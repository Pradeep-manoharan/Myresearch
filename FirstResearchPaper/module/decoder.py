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



# Outputprocess
#(i) Processes
class OutputProcess(AbstractProcess):
    def __init__(self, num_step, **kwargs):
        super().__init__()
        n_img = kwargs.pop("n_img", 25)
        self.in_port = InPort(shape=(10,))
        self.label_in = InPort(shape=(1,))

        self.n_step_per_image = Var(shape=(1,), init=num_step)
        self.n_img = Var(shape=(1,), init=n_img)
        self.spike_accoumulate = Var(shape=(10,))
        self.predicted_labels = Var(shape=(n_img,))
        self.ground_truth_labels = Var(shape=(n_img,))

# (ii) ProcessModel
@implements(proc=OutputProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class OutputProcessModel(PyLoihiProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    label_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=32)
    n_step_per_image: int = LavaPyType(int, int, precision=32)
    n_img: int = LavaPyType(int, int, precision=32)
    spike_accoumulate: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    predicted_labels: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    ground_truth_labels: np.ndarray = LavaPyType(np.ndarray, int, precision=32)

    def __init__(self, proc=None):
        super().__init__(proc)
        self.current_img_id = 0

    def run_spk(self):
        spk_in = self.in_port.recv()
        lbl_in = self.label_in.recv()

        self.spike_accoumulate = spk_in + self.spike_accoumulate

        # if self.time_step % 500  == 0:
        #     print(f'Ground label received in the output process {lbl_in}')
        #     print(f'Spike in the output process {spk_in}')
        

        #print(f'Let see the time step {self.time_step} and n_step_per_img {self.n_step_per_image}')
        if self.time_step % self.n_step_per_image == 0 and self.time_step > 1:

            self.predicted_labels[self.current_img_id] = np.argmax(self.spike_accoumulate)
            self.ground_truth_labels[self.current_img_id] = lbl_in
            self.current_img_id += 1
            if self.time_step % 100 == 0:
                print(f' Output time step : {self.time_step} Spike  Output spike : {self.spike_accoumulate} and  Label : {lbl_in}')
            self.spike_accoumulate = np.zeros_like(self.spike_accoumulate)
            
