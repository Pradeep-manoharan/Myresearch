# 1) Project Overview 
"""
The ultimate goal of the project is to build a Spike Neural Network using the MNIST dataset,
train it, visualize the result. 

In this project we gonna first convert MNIST dataset into spike trains,feed them into the network, train the model using STDP learning rule.
The training process and result will be visualized to better understand the learning dynamics.

Let's deep delve the network architecture, here we have three process model, first one will be spikeinput which help to convert MNIST dataset into spike trains
This will be connect to image classification model which spike out for each image and final process is outputprocess which is generate the output 

Let's build the architecture;

[Input Data] --> [SpikeTrain --> LIF --> LearningDense(STDP) ---> LIF] ---> [OutProcess]
  SpikeInput Proceses                       Image classification              Output
"""

# 2) Setup & Import dependency
# Imports for Processes
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

# Imports for ProcessModel
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.type import LavaPyType
from lava.proc.lif.process import LIF

# Imports for learning Rule
from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi
from lava.proc.dense.process import LearningDense

# Imports for Run Simulation 
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg

# Monitor
from lava.proc.monitor.process import Monitor

# Import Dataset
from lava.utils.dataloader.mnist import MnistDataset

# Imports fundamental library
import numpy as np
import matplotlib.pyplot as plt


'As far now, we have import required libray, Let s build Spike Input'

# 3) Build the Proceses and his ProcesModel
# (i) SpikeInput Proceses
class SpikeInput(AbstractProcess):
    def __init__(self, vth = 1, num_image = 25 , step_per_image = 128):
        super().__init__()
        shape = (784,)

        # Let's define the Ports
        self.Spike_out = OutPort(shape = shape)
        self.label_out = OutPort(shape = (1,))

        # Let's define the Var Port
        self.num_image = Var(shape = (1,), init = num_image)
        self.num_step_image =  Var(shape = (1,), init = step_per_image)
        self.input_img = Var(shape = shape)
        self.ground_truth_label = Var(shape = (1,))
        self.v = Var(shape = shape, init = 0)
        self.vth = Var(shape = (1,), init = vth)

@implements(proc = SpikeInput, protocol = LoihiProtocol)    
@requires(CPU)
class SPikeInputBehavior(PyLoihiProcessModel):
    Spike_out : PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool,precision = 1)
    label_out : PyOutPort = LavaPyType(PyOutPort.VEC_DENSE,np.int32,precision = 32)

    num_image : int = LavaPyType(int,int,precision = 32)
    num_step_image : int = LavaPyType(int, int,precision = 32)
    input_img : np.ndarray = LavaPyType(np.ndarray, int, precision = 32)
    ground_truth_label : int = LavaPyType(int, int, precision = 32)
    v : np.ndarray = LavaPyType(np.ndarray,int, precision  = 32)
    vth : np.ndarray = LavaPyType(int, int, precision = 32)

    def __init__(self,proc= None):
        super().__init__(proc)
        self.dataset = MnistDataset()
        self.currect_img_id = 0

    def post_guard(self,):
        if self.time_step % self.num_step_image == 1:
            return True
        return False
    
    def run_post_mgmt(self):
        img = self.dataset.images[self.currect_img_id]
        self.ground_truth_label = self.dataset.labels[self.currect_img_id]
        self.input_img = img.astype(np.int32) - 127
        self.v = np.zeros(self.v.shape)
        self.currect_img_id += 1

    def run_spk(self):
        self.v[:] = self.v + self.input_img
        s_out = self.v > self.vth
        self.v[s_out] = 0
        label_out = np.array([self.ground_truth_label],dtype = np.int32)
        # print(f'Output shape {s_out.shape}')
        # print(f'label out {label_out.shape}')
        self.Spike_out.send(s_out)
        self.label_out.send(label_out)


'''
As so for we build spike input object. here after we gonna build neural archicture which is capable learn
weights between neurons. To do this we gonna intergrate STDP Module inside the architecture. 

[Input Data] --> [SpikeTrain --> LIF --> LearningDense(STDP) ---> LIF] ---> [OutProcess]
  SpikeInput Proceses                       Image classification              Output
'''
#(ii) Image Classification 

class ImageClassifier(AbstractProcess):
    def __init__(self,):
        super().__init__()

        # Let's initate random weight and bias    
        w1 = np.random.rand(10,784) * 5
        b1 = np.random.rand(784,)
        b2 = np.random.rand(10,)
        
        self.spike_in = InPort(shape = (784,)) 
        self.b_lif1 = Var(shape = (784,),init = b1)
        self.learning_dense = Var(shape = w1.shape, init = w1)
        self.b_lif2  = Var(shape = (10,),init = b2)
        self.spike_out = OutPort(shape = (10,))

        # Up-level currents and voltage of LIF Processes
        self.lif1_u = Var(shape = (784,), init = 0)
        self.lif1_v = Var(shape = (784,), init = 0)
        self.lif2_u = Var(shape = (10,), init = 0)
        self.lif2_v = Var(shape = (10,), init = 0)

        

@implements(proc = ImageClassifier, protocol = LoihiProtocol)
@requires(CPU)
class ImageClassifierBehavior(AbstractSubProcessModel):
    stdp = STDPLoihi(learning_rate=1,
                 A_plus=1,
                 A_minus=-1,
                 tau_plus=10,
                 tau_minus=10,
                 t_epoch=4)
    def __init__(self,proc):
        
        self.lif1 = LIF(shape = (784,),bias_mant= proc.b_lif1.init, vth = 1,dv = 0,du = 1)        
        self.plastic_connection =  LearningDense(weights = proc.learning_dense.init,learning_rule = self.stdp,name = "plastic_dense")
        self.lif2 = LIF(shape = (10,), bias_mant = proc.b_lif2.init, vth = 1, dv = 0, du = 1)

        proc.spike_in.connect(self.lif1.a_in)
        self.lif1.s_out.connect(self.plastic_connection.s_in)
        self.plastic_connection.a_out.connect(self.lif2.a_in)
        self.lif2.s_out.connect(proc.spike_out)


        self.lif2.s_out.connect(self.plastic_connection.s_in_bap)

        proc.lif1_u.alias(self.lif1.u)
        proc.lif1_v.alias(self.lif1.v)
        proc.lif2_u.alias(self.lif2.u)
        proc.lif2_v.alias(self.lif2.v)

"""
As so far, we have implement the spike input and imageclassification Process and ProcssModel. 
Now we gonna implement the output processes and its process Model
"""

class OutProcess(AbstractProcess):
    def __init__(self,**kwargs):
        super().__init__()

        num_image = kwargs.pop("n_img",25)
        self.spike_in = InPort(shape = (10,))
        self.label_in = InPort(shape = (1,))

        self.num_image = Var(shape = (1,), init = num_image)
        self.num_step_image = Var(shape = (1,), init = 128)
        self.spike_accoumulate = Var(shape = (10,))
        self.predict_label = Var(shape = (num_image,))
        self.gt_labels = Var(shape = (num_image,))



@implements(proc = OutProcess, protocol = LoihiProtocol)
@requires(CPU)
class OutputProcessBehavior(PyLoihiProcessModel):
    spike_in = LavaPyType(PyInPort.VEC_DENSE, int, precision=32)
    label_in = LavaPyType(PyInPort.VEC_DENSE, int, precision=32)

    num_image = LavaPyType(int, int, precision=32)
    num_step_image: int = LavaPyType(int, int, precision=32)
    spike_accoumulate: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    predict_label = LavaPyType(np.ndarray, int, precision=32)
    gt_labels = LavaPyType(np.ndarray, int, precision=32)

    def __init__(self, proc_params=None):
        super().__init__(proc_params=proc_params)
        self.currect_img_id = 0


    def post_guard(self):
        if self.time_step % self.num_step_image == 0 and self.time_step > 0 :
            return True
        return False
    
    def run_post_mgmt(self):
        pass      
        


    def run_spk(self):
        spk_in = self.spike_in.recv()
        gt_label = self.label_in.recv()
        
        pred_label = np.argmax(self.spike_accoumulate)
        self.predict_label[self.currect_img_id] = pred_label
        self.gt_labels[self.currect_img_id] = gt_label
        print(f'The step time {self.time_step} and spike is {spk_in}, label is {gt_label} and Predict Label is {pred_label}')
        self.spike_accoumulate = spk_in + self.spike_accoumulate

        if self.time_step % self.num_step_image == 0 and self.time_step > 0:
            self.currect_img_id += 1
            self.spike_accoumulate = np.zeros_like(self.spike_accoumulate)
        




# Let's define the function for visualization

def visualize_spikes_and_voltage(spike_input,output_spike,lif1_voltage,lif2_voltage,num_images,steps_per_image):
    """
    Let's visualize the spike and voltage of network
    """

    # Let's caputure the keys
    spike_input_key = list(spike_input.keys())
    output_key = list(output_spike.keys())
    lif1_key = list(lif1_voltage.keys())
    lif2_key = list(lif2_voltage.keys())


    #Extract spike data
    input_spike_data = spike_input[spike_input_key[0]]['Spike_out']
    output_spike_data = output_spike[output_key[0]]["spike_out"]
    lif1_voltage_data = lif1_voltage[lif1_key[0]]["lif1_v"]
    lif2_voltage_data = lif2_voltage[lif2_key[0]]["lif2_v"]

    # Let's create subplot

    fig, axes = plt.subplots(2,2,figsize = (15,10))
    fig.suptitle("Spiking Network visualization", fontsize = 16)    

    axes1 = axes[0,0]

    # Let's create the raster plot
    print(len(input_spike_data[0]))
    spike_time, spike_neuron = np.where(input_spike_data)
    
    if len(spike_time) > 0:
        axes1.scatter(spike_time,spike_neuron,s = 1, c = "blue",alpha = 1)
  
    else:
        axes1.text(0.5,0.5,"No Input Spike dectected",ha = "center",va = "center",transform = axes1.transAxes)

    axes1.set_xlabel("Time Steps")
    axes1.set_ylabel("Neuron Index")
    axes1.set_title("Input Spike Raster")
    axes1.grid(True,alpha = 0.3)

    for i in range(1,num_images):
        axes1.axvline(x = i* steps_per_image,color = "Red",linestyle = "--",alpha = 0.5)

    axes2 = axes[0,1]

    spike_time, spike_neuron = np.where(output_spike_data)
    if len(spike_time) > 0:
        axes2.scatter(spike_time,spike_neuron)

    else:
        axes2.text(0.5,0.5,"No Input Spike dectected",ha = "center",va = "center")

    axes2.set_xlabel("Time step")
    axes2.set_ylabel("Neuron Index")
    axes2.set_title("Image Classification Output Spike")
    axes2.grid(True,alpha = 0.5)


    plt.show()


if __name__ == "__main__":
    # General Parameters
    num_image = 200
    step_per_image = 5
    
    numstep = num_image * step_per_image
    

    # Instatiate the class    
    spike = SpikeInput(vth = 1, num_image = num_image, step_per_image = step_per_image)
    image_classifier = ImageClassifier()
    outputprocess = OutProcess(n_img = num_image)

    # Connection

    spike.Spike_out.connect(image_classifier.spike_in)
    image_classifier.spike_out.connect(outputprocess.spike_in)
    spike.label_out.connect(outputprocess.label_in)

    # Create Monitor
    spike_input_monitor = Monitor()
    imageclassifier_monitor = Monitor()
    lif1_v_monitor = Monitor()
    lif2_v_monitor = Monitor()      



    # Run tge simulation
    try:
        # Add Probe
        spike_input_monitor.probe(spike.Spike_out, num_steps = numstep,)
        imageclassifier_monitor.probe(image_classifier.spike_out,num_steps = numstep)
        lif1_v_monitor.probe(image_classifier.lif1_v,num_steps = numstep)
        lif2_v_monitor.probe(image_classifier.lif2_v,num_steps = numstep)

        # Run the simulation
        spike.run(condition = RunSteps(num_steps = numstep),run_cfg= Loihi2SimCfg(select_sub_proc_model = True, select_tag='fixed_pt',))

        #Get the data
        spike_input = spike_input_monitor.get_data()
        image_classifier_output = imageclassifier_monitor.get_data()
        lif1_v_monitor_output = lif1_v_monitor.get_data()
        lif2_v_monitor_output = lif2_v_monitor.get_data()

        visualize_spikes_and_voltage(spike_input,image_classifier_output,lif1_v_monitor_output,
                                     lif2_v_monitor_output,num_images = num_image,steps_per_image = step_per_image)


      
        spike.stop()
        print("Simulation completed sucessfully")

    except Exception as e:
        print(f'Simulation error  {e}')
        import traceback
        traceback.print_exc()