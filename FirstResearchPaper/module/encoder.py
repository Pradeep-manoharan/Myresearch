
# Setup the enviroment and import requrire library
# Gentral
import numpy as np
import matplotlib.pyplot as plt
import typing as type
import lava
import sys

# Processes
from lava.proc.lif.process import LIF,LIFReset
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
import pickle



# Build the encoder
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
@tag("floating_pt")
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
        img = self.dataset.images[0]
        self.ground_truth_label = self.dataset.labels[0]
        self.v = np.zeros_like(self.v)
        self.image = img.astype(np.int32) -127
        self.curret_image_id = 1

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
        self.v[s_out] = 0  # Reset voltage of neurons that spiked          
        label_out = np.array([self.ground_truth_label],dtype = np.int32)
        self.outport.send(s_out)
        self.label_out.send(label_out)
