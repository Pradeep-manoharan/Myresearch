# 1) Project Overview

"""
The ultimate goal of this project is to build an SNN model using the MNIST dataset with the Lava framework by Intel Loihi. 
The neural network will be designed to accommodate real-time learning with predictive capabilities, without being excessively deep or shallow. 
We will evaluate the model's accuracy and compare it with state-of-the-art models such as Diehl & Cook (Accuracy ~95%) and a simple ANN (Accuracy ~97-98%).

Process Structure:

Encoder → Image Classification → Output Process → Evaluation Process → Visualization Process

1) Encoder Process
The encoder will convert the MNIST dataset into spike trains and send them to the image classification process.

2) Image Classification Process
This process may have multiple layers with custom learning rules or STDP learning rules.

3) Output Process
This process will take the spikes from image classification and convert them into output values.

4) Evaluation Process
This process will compare the predicted labels with the true labels and generate accuracy metrics.

5) Visualization Process
This process will visualize the output for each batch or iteration, showing how accuracy changes over time.

Note: The model should be as dynamic as possible to allow customization of the architecture and hyperparameters for future tuning.

"""

# 2) Setup the enviroment and import requrire library
# Gentral
import numpy as np
import matplotlib.pyplot as plt
import typing as type
import lava

# Processes
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense, LearningDense
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

# ProcessModel
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.decorator import implements, tag, requires
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

# Learning Rule
from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi

# Running
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import RunConfig, Loihi2SimCfg

# Monitor
from lava.proc.monitor.process import Monitor

# Dataset
from  lava.utils.dataloader.mnist import MnistDataset
from lava.magma.core.run_configs import Loihi2SimCfg


# 3) Build the encoder
# Encoder Processes
class Encoder(AbstractProcess):
    """
    This procesess helps to convert MNIST data into spike train 
    """
    def __init__(self,n_img = 25, num_step_per_image = 128,vth = 1):
        super().__init__()
        shape = (784,)
        self.outport = OutPort(shape = shape)
        self.label_out = OutPort(shape = (1,))

        self.n_img = Var(shape = (1,),init = n_img)
        self.num_step_per_image = Var(shape = (1,), init = num_step_per_image)
        self.vth = Var(shape = (1,),init = vth)
        self.ground_truth_label = Var(shape = (1,))
        self.image = Var(shape = shape)
        self.v = Var(shape = shape,init = 0)
        
# Encoder ProcessModel
@implements(proc = Encoder,protocol = LoihiProtocol )
@requires(CPU)
@tag("fixed_pt")
class EncoderProcessModel(PyLoihiProcessModel):

    outport : PyOutPort= LavaPyType(PyOutPort.VEC_DENSE,bool,precision = 1)
    label_out : PyOutPort = LavaPyType(PyOutPort.VEC_DENSE,int,precision = 32)
    n_img = LavaPyType(int,int,precision = 32)
    num_step_per_image : int = LavaPyType(int,int,precision = 32)
    vth = LavaPyType(int,int, precision = 32)
    v :np.ndarray = LavaPyType(np.ndarray, int, precision = 32)
    ground_truth_label = LavaPyType(int,int,precision = 32)
    image = LavaPyType(np.ndarray,int,precision = 32)


    def __init__(self,proc = None):
        super().__init__(proc)
        self.dataset = MnistDataset()
        self.curret_image_id = 0

    def post_guard(self,):
        if self.time_step % self.num_step_per_image == 0:
            return True
        return False    

    def run_post_mgmt(self,):
        img = self.dataset.images[self.curret_image_id]
        self.ground_truth_label = self.dataset.labels[self.curret_image_id]
        self.image = img.astype(np.int32) -127
        self.v = np.zeros(self.v.shape)
        self.curret_image_id += 1

    def run_spk(self,):
        self.v[:] = self.v + self.image
        s_out = self.v > self.vth
        if self.time_step % 10 == 0:
            print(f"Let's see the time step {self.time_step} and spike {s_out.shape}" )
        label_out = np.array([self.ground_truth_label],dtype = np.int32)
        self.outport.send(s_out)
        self.label_out.send(label_out)


# 4) Build the imageclassification procesess
# (i) The Processes for imageclassification
class Imageclassification(AbstractProcess):
    """
    This class defines the architecture of the layers. 
    It is designed to be dynamic, allowing easy adjustment of the number of layers and neurons.
    For training, this layer will use STDP learning rules with a learning dense layer.
    """

    def __init__(self,layer_sizes = None):
        """
        Args: 
        Layer_size (list of int) : List of specifying the number of neuron in each LIF
                                   Example = [784,128,64,10]
        """
        super().__init__()

        self.spike_in = InPort(shape = (784,))
        self.spike_out = OutPort(shape = (10,))

        if layer_sizes == None:
            self.layer_sizes = [784,128,10] # Default : Input, Hidden, Output
        else:
            self.layer_sizes = layer_sizes
        self.hidden_layers = []
        self.hidden_weights = []
        self.alias_var_u = []
        self.alias_var_v = []
        #self.alias_var_s_out = []
       

        for i in range(len(self.layer_sizes)):            
                self.hidden_layers.append(Var(shape = (self.layer_sizes[i],),init = (np.random.rand(self.layer_sizes[i],)),name = f"Layer_{i}"))
                # Create alias for monitor the voltage and current
                u_var = Var(shape = (self.layer_sizes[i],),init = 0,name = f'lif{i}_u')
                v_var = Var(shape = (self.layer_sizes[i],),init = 0,name = f'lif{i}_v')
                s_var = Var(shape = (self.layer_sizes[i],),init = 0,name = f'lif{i}_s_out')
                setattr(self,f'lif{i}_u',u_var)
                setattr(self,f'lif{i}_v',v_var)
                setattr(self,f'lif{i}_s_out',s_var)
                self.alias_var_u.append(u_var)
                self.alias_var_v.append(v_var)
                #self.alias_var_s_out.append(s_var)
                

                if i < len(self.layer_sizes)-1:
                    self.hidden_weights.append(Var(shape = (self.layer_sizes[i],self.layer_sizes[i+1]),init = (np.random.rand(self.layer_sizes[i],self.layer_sizes[i+1])), name = f'Plastic_layer_{i}'))


      

# (ii) Processes Model for imageclassification
@implements(Imageclassification)
@requires(CPU)
@tag("fixed_pt")
class ImageclassificationProcessModel(AbstractSubProcessModel):     
        def __init__(self, proc):
            self.layer_sizes = proc.layer_sizes
            self.hidden_layers = proc.hidden_layers
            self.hidden_weights = proc.hidden_weights
            self.alias_var_u = proc.alias_var_u
            self.alias_var_v = proc.alias_var_v
            #self.alias_var_s_out = proc.alias_var_s_out
            self.lif_layers = []    
            self.plastic_layers = []

            print("Building the layer")
            for i in range(len(self.layer_sizes)):                
                lif = LIF(shape = (self.layer_sizes[i],),bias_mant = self.hidden_layers[i].init,dv = 0,du = 4094,name  = f'lif_{i}',vth = 10)  
                setattr(self,f"lif_{i}",lif)              
                self.lif_layers.append(lif)
                print(f'Append LIF layer {i}: shape= {self.hidden_layers[i].shape} ')
                if i < len(self.layer_sizes)-1: #[0,1,2,3,4]
                    id = LearningDense(
                                        weights = self.hidden_weights[i].init.T, 
                                        learning_rule= STDPLoihi(learning_rate=1,
                                                                A_plus=1,
                                                                A_minus=-1,
                                                                tau_plus=10,
                                                                tau_minus=10,
                                                                t_epoch=4),
                                        name = f'plasity_{i}'
                                        )
                    self.plastic_layers.append(id)
                    setattr(self,f'plasity_{i}',id)
                    print(f'Append LearningDense Layer {i}: shape = {self.hidden_weights[i].shape}')
            print()
                    
            # Connect input to first LIF
            proc.spike_in.connect(self.lif_layers[0].a_in)

            # Connect layers dynamically
            for i in range(len(self.layer_sizes)-1): #[784,128,10]
                self.lif_layers[i].s_out.connect(self.plastic_layers[i].s_in)  #(784), (784,128)
                print(f'Connection is made between {self.lif_layers[i].name} and {self.plastic_layers[i].name}')
                self.plastic_layers[i].a_out.connect(self.lif_layers[i+1].a_in)
                print(f'Connection is made between {self.plastic_layers[i].name} and {self.lif_layers[i+1].name}')
                if hasattr(self.plastic_layers[i],"s_in_bap"):
                    self.lif_layers[i+1].s_out.connect(self.plastic_layers[i].s_in_bap)
                    print(f'The BAP connection also made between {self.lif_layers[i+1].name} and {self.plastic_layers[i].name}')
            print()
            self.lif_layers[-1].s_out.connect(proc.spike_out)

            self.alias_u_list = []
            self.alias_v_list = []
            self.alias_s_out_list  =[]
            # Let's create alias
            for i, lif in enumerate(self.lif_layers):
                self.alias_u_list.append(self.alias_var_u[i].alias(lif.u))
                self.alias_v_list.append(self.alias_var_v[i].alias(lif.v))
                #print(lif.s_out)
            
                #self.alias_s_out_list.append(self.alias_var_s_out[i].alias(lif.s_out))




    




# 5) Outputprocess
#(i) Processes
class OutputProcess(AbstractProcess):
    def __init__(self,num_step,**kwargs):
        super().__init__()
        n_img =  kwargs.pop("n_img",25)
        self.spike_in = InPort(shape = (10,))
        self.label_in = InPort(shape = (1,))

        self.n_step_per_image = Var(shape = (1,), init = num_step)
        self.n_img = Var(shape = (1,), init = n_img)
        self.spike_accoumulate = Var(shape = (10,))
        self.pred_label = Var(shape = (n_img,))
        self.gred_label = Var(shape = (n_img,))

# (ii) ProcessModel
@implements(proc = OutputProcess,protocol = LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class OutputProcessModel(PyLoihiProcessModel):
    spike_in = LavaPyType(PyInPort.VEC_DENSE,bool,precision = 1)
    label_in = LavaPyType(PyInPort.VEC_DENSE,int,precision = 1)

    n_step_per_image : int = LavaPyType(int,int,precision = 32)
    n_img = LavaPyType(int,int,precision = 32)
    spike_accoumulate = LavaPyType(np.ndarray, int, precision = 32)
    pred_label : np.ndarray  = LavaPyType(np.ndarray,int,precision = 32)
    gred_label : np.ndarray = LavaPyType(np.ndarray,int, precision = 32)

    def __init__(self,proc = None):
        super().__init__(None)
        self.currect_img_id = 0

    def run_spk(self):
        spk_in = self.spike_in.recv()
        lbl_in = self.label_in.recv()
        if self.time_step % 10  == 0:
            print(f'Ground label received in the output process {lbl_in}')
            print(f'Spike in the output process {spk_in}')
        

        self.spike_accoumulate = spk_in + self.spike_accoumulate
        #print(f'Let see the time step {self.time_step} and n_step_per_img {self.n_step_per_image}')
        if self.time_step % self.n_step_per_image == 0 and self.time_step > 1:
            self.pred_label[self.currect_img_id] = np.argmax(self.spike_accoumulate)
            self.gred_label[self.currect_img_id] = lbl_in
            self.currect_img_id += 1
            self.spike_accoumulate = np.zeros_like(self.spike_accoumulate)



if __name__ == "__main__":
    # General Parameters
    num_image = 10
    step_per_image = 5
    num_step = num_image * step_per_image

    # Processes
    encoder = Encoder(n_img = num_image, num_step_per_image = step_per_image,vth = 1)
    image_classification = Imageclassification(layer_sizes = [784,500,128,10])
    decoder = OutputProcess(num_step=step_per_image,n_img = num_image)

    

    # Connection
    encoder.outport.connect(image_classification.spike_in)
    image_classification.spike_out.connect(decoder.spike_in)
    encoder.label_out.connect(decoder.label_in)
   

    # Monitor
    # Create Monitor for LIF
    monitor_u_layer_1 = Monitor()
    monitor_v_layer_1  = Monitor()
    monitor_spike_layer_1= Monitor()

    

    # Connect the monitor to ImageClassification Process
    monitor_u_layer_1.probe(target = image_classification.alias_var_u[0], num_steps = num_step)
    monitor_v_layer_1.probe(target = image_classification.alias_var_v[0], num_steps = num_step)
    #monitor_spike_layer_1.probe(target = image_classification.alias_var_s_out[0],num_steps =num_step)



    # Running:
    try:                
        encoder.run(condition = RunSteps(num_steps = num_step),
                    run_cfg = Loihi2SimCfg(select_sub_proc_model = True))
                    
        monitor_u_layer = monitor_u_layer_1.get_data()["Process_1"]["lif0_u"]
        monitor_v_layer = monitor_v_layer_1.get_data()["Process_1"]["lif0_v"]

                    
        ground_truth_label = decoder.gred_label.get().astype(np.int32)
        prediction_label = decoder.pred_label.get().astype(np.int32)
        encoder.stop()

        print(monitor_u_layer.shape)
        print(monitor_v_layer.shape)

        plt.figure(figsize= (10,8))
        plt.subplot(2,1,1)
        plt.plot(monitor_v_layer[:,:50])
        print(monitor_v_layer[:,:50])
        plt.title("Membrance Potential for first five neuron")
        plt.xlabel("Time step")
        plt.ylabel("Potential")
        plt.legend([f'Neuron{i}' for i in range(50)])
        plt.show()
        

        

        Accuracy = np.sum(ground_truth_label == prediction_label) / ground_truth_label.size * 100
        print(f'\nGround Truth label {ground_truth_label}\n'
              f'\nPrediction lable {prediction_label}\n'
              f'\nAccuracy {Accuracy}')
    except Exception as e:
        print()
        print(f'Simulation error {e}')
        print()
        import traceback 
        traceback.print_exc()
