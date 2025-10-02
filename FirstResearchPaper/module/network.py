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


# Define the network function for all layer

def build_network(layer_sizes, dv_vals, du_vals, vth, bias_mant, learning_rate, a_plus, a_minus, tau_plus, tau_minus, t_epoch, use_lateral_inhibition :bool = False):
    """Builds the LIF and LearningDense layers for the network."""
    lif_layers = []
    plastic_layers = []

    print("Building Layer")

    for i, size in enumerate(layer_sizes):
        lif = LIF(shape=(size,),
                  dv=dv_vals[i],
                  du=du_vals[i],
                  name=f'lif_{i}',
                  vth=vth[i],
                  bias_mant=bias_mant)
        lif_layers.append(lif)
        print(f'Append LIF layer {i}: shape = {size},')

        if i < len(layer_sizes) - 1:
            weights = np.random.normal(0.5, 0.1, (layer_sizes[i], layer_sizes[i+1]))
            weights = np.clip(weights, 0, 1)

            ld = LearningDense(
                weights=weights.T,
                learning_rule=STDPLoihi(learning_rate=learning_rate,
                                        A_plus=a_plus,
                                        A_minus=a_minus,
                                        tau_plus=tau_plus,
                                        tau_minus=tau_minus,
                                        t_epoch=t_epoch),
                name=f'plastic_{i}'
            )

            plastic_layers.append(ld)
            print(f'Append LearningDense Layer {i}: size = ({layer_sizes[i], layer_sizes[i+1]})')

        # Add Lateral Inhibition to hidden and output layers
        if use_lateral_inhibition:
            print("\nLateral Inhibition is Enabled")
            for i in range(1, len(lif_layers)):
                layer_to_inhibit = lif_layers[i]
                layer_size = layer_sizes[i]

                # Create a weight matrix where each neuron inhibits all others but not itself.
                inhib_weight = -1 * (np.ones((layer_size,layer_size)) - np.identity(layer_size))

                # Create a non-learning Dense process for the inhibitory connections
                inhib_layer = Dense(weights = inhib_weight, name = f'Inhib_{i}')

                # Create a layer to the inhibitory process and bact to itself
                layer_to_inhibit.s_out.connect(inhib_layer.s_in)
                inhib_layer.a_out.connect(layer_to_inhibit.a_in)
                print(f"Winnder-Taker-All circuit added to '{layer_to_inhibit.name}'")
        else:
            print("\nLateral Inhibition is DISABLED")

    return lif_layers, plastic_layers
