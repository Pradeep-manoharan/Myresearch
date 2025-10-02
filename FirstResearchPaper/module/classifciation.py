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

# #4) Build the imageclassification procesess
# #(i) The Processes for imageclassification
# class Imageclassification(AbstractProcess):
#     """
#     This class defines the architecture of the layers. 
#     It is designed to be dynamic, allowing easy adjustment of the number of layers and neurons.
#     For training, this layer will use STDP learning rules with a learning dense layer.
#     """

#     def __init__(self,layer_sizes = None,
#                   v_track : bool = True,
#                     u_track : bool = True, 
#                     w_track : bool = True,
#                     s_track :bool = True,
#                     monitor_step : int = 0):
#         """
#         Args: 
#         Layer_size (list of int) : List of specifying the number of neuron in each LIF
#                                    Example = [784,128,64,10]
#         """
#         super().__init__()

#         self.spike_in = InPort(shape = (784,))
#         self.spike_out = OutPort(shape = (10,))

#         if layer_sizes == None:
#             self.layer_sizes = [784,128,10] # Default : Input, Hidden, Output
#         else:
#             self.layer_sizes = layer_sizes
#         self.hidden_layers = []
#         self.hidden_weights = []

#         # Monitor Control read by ProcessModel
#         self.monitor_steps = Var(shape = (1,), init = int(monitor_step), name = "monitor_step")
#         self.track_v = Var(shape = (1,),init = int(v_track),name = "track_v")
#         self.track_u = Var(shape = (1,),init = int(u_track),name = "track_u")
#         self.track_w = Var(shape = (1,),init = int(w_track),name = "track_w")
#         self.track_s = Var(shape = (1,),init = int(s_track),name = "track_s")

      
#        # Build layer placeholders and random inits
#         for i in range(len(self.layer_sizes)):            
#                 self.hidden_layers.append(Var(shape = (self.layer_sizes[i],),init = (np.random.rand(self.layer_sizes[i],)),name = f"Layer_{i}"))

#                 if i < len(self.layer_sizes)-1:
#                     self.hidden_weights.append(Var(shape = (self.layer_sizes[i],self.layer_sizes[i+1]),init = (np.random.rand(self.layer_sizes[i],self.layer_sizes[i+1])), name = f'Plastic_layer_{i}'))

    

# # (ii) Processes Model for imageclassification
# @implements(Imageclassification)
# @requires(CPU)
# @tag("fixed_pt")
# class ImageclassificationProcessModel(AbstractSubProcessModel):     
#         def __init__(self, proc):
#             self.layer_sizes = proc.layer_sizes
#             self.hidden_layers = proc.hidden_layers
#             self.hidden_weights = proc.hidden_weights
  
#             self.lif_layers = []    
#             self.plastic_layers = []

#             print("Building the layer")
#             for i in range(len(self.layer_sizes)):                
#                 lif = LIF(shape = (self.layer_sizes[i],),bias_mant = self.hidden_layers[i].init,dv = 0,du = 4094,name  = f'lif_{i}',vth = 10)  
#                 setattr(self,f"lif_{i}",lif)              
#                 self.lif_layers.append(lif)
#                 print(f'Append LIF layer {i}: shape= {self.hidden_layers[i].shape} ')
#                 if i < len(self.layer_sizes)-1: #[0,1,2,3,4]
#                     id = LearningDense(
#                                         weights = self.hidden_weights[i].init.T, 
#                                         learning_rule= STDPLoihi(learning_rate=1,
#                                                                 A_plus=1,
#                                                                 A_minus=-1,
#                                                                 tau_plus=10,
#                                                                 tau_minus=10,
#                                                                 t_epoch=4),
#                                         name = f'plasity_{i}'
#                                         )
#                     self.plastic_layers.append(id)
#                     setattr(self,f'plasity_{i}',id)
#                     print(f'Append LearningDense Layer {i}: shape = {self.hidden_weights[i].shape}')
#             print()
                    
#             # Connect input to first LIF
#             proc.spike_in.connect(self.lif_layers[0].a_in)

#             # Connect layers dynamically
#             for i in range(len(self.layer_sizes)-1): #[784,128,10]
#                 self.lif_layers[i].s_out.connect(self.plastic_layers[i].s_in)  #(784), (784,128)
#                 print(f'Connection is made between {self.lif_layers[i].name} and {self.plastic_layers[i].name}')
#                 self.plastic_layers[i].a_out.connect(self.lif_layers[i+1].a_in)
#                 print(f'Connection is made between {self.plastic_layers[i].name} and {self.lif_layers[i+1].name}')
#                 if hasattr(self.plastic_layers[i],"s_in_bap"):
#                     self.lif_layers[i+1].s_out.connect(self.plastic_layers[i].s_in_bap)
#                     print(f'The BAP connection also made between {self.lif_layers[i+1].name} and {self.plastic_layers[i].name}')
#             print()
#             self.lif_layers[-1].s_out.connect(proc.spike_out)

#             #Create the attach monitors here
#             steps= int(proc.monitor_steps.init)
#             track_v = bool(proc.track_v.init)
#             track_u  = bool(proc.track_u.init)
#             track_s = bool(proc.track_s.init)
#             track_w = bool(proc.track_w.init)

#             mon_v, mon_u, mon_s, mon_w = [],[],[],[]
            
#             for lif in self.lif_layers:
#                 print(lif.name)
#                 if track_v:
#                     mv = Monitor(); mv.probe(target = lif.v, num_steps = steps); mon_v.append(mv)
#                 if track_u:
#                     mu = Monitor(); mu.probe(target = lif.u, num_steps = steps); mon_u.append(mu)
#                 if track_s:
#                     ms = Monitor(); ms.probe(target = lif.s_out, num_steps = steps); mon_s.append(ms)

#             if track_w:

#                 for dense in self.plastic_layers:
#                     mw = Monitor(); mw.probe(target = dense.weights, num_steps = steps);mon_w.append(mw)

#             #Expose monitor to parent class(process) to read the data
#             setattr(proc,"mon_v", mon_v)
#             setattr(proc,"mon_u", mon_u)            
#             setattr(proc,"mon_s", mon_s)
#             setattr(proc,"mon_w", mon_w)
 