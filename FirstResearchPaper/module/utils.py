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



class WeightNormalizer(AbstractProcess):
    """A process to periodically normalize weights of a target layer."""
    def __init__(self, target_weights: Var, interval: int,name : str):
        super().__init__(name = name)
        self.weights = target_weights
        self.interval = Var(shape=(1,), init=interval)
        self.trigger_in = InPort(shape=(1,))
       

@implements(proc=WeightNormalizer, protocol=LoihiProtocol)
@requires(CPU)
class PyWeightNormalizerModel(PyLoihiProcessModel):
    weights: np.ndarray = LavaPyType(np.ndarray, float)
    interval: int = LavaPyType(int, int)
    trigger_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    
    def __init__(self,proc_params):
        super().__init__(proc_params)
        self.name = self.proc_params.get("name","DefaultName")


    def post_guard(self):
        return (self.time_step % self.interval == 0) and (self.time_step > 0)

    def run_post_mgmt(self):
      
        w = self.weights
        norm = np.sqrt(np.sum(w**2, axis=0, keepdims=True))
        norm = np.maximum(norm, 1e-5)
        self.weights[:] = w / norm
        print(f"| -- Weights normalized at timestep {self.time_step} -- |")

    def run_spk(self):
        self.trigger_in.recv()
        # if self.time_step % self.interval == 0 and self.time_step > 0 :            
        #     w = self.weights
        #     norm = np.sqrt(np.sum(w**2, axis  = 0, keepdims  = True))
        #     norm  = np.maximum(norm, 1e-5)
        #     self.weights[:] = w / norm
        #     print(f'|-- Weights Normalized at time-step {self.time_step} and {self.name} --|')


def plot_neuron_behaviour(v_data, neuron_count_visual, time_count_visual, u_data=None, s_data=None, initial_weights=None, final_weights=None, lif_layer=None, plastic_layers=None, learning_params=None, visualize_weights=True):
    """
    Plots voltage, current, and spike raster plot for each layer in the network.
    Also plots heatmaps of initial and final weights if requested.

    Args:
        v_data (list): List of voltage data arrays for each layer.
        neuron_count_visual (int): Number of neurons to visualize.
        time_count_visual (int): Number of time steps to visualize.
        u_data (list, optional): List of current data arrays for each layer.
        s_data (list, optional): List of spike data arrays for each layer.
        initial_weights (list, optional): List of initial weight matrices.
        final_weights (list, optional): List of final weight matrices.
        lif_layer (list, optional): List of the network's LIF layer process.
        plastic_layers (list, optional): List of the network's plastic layer process.
        learning_params (dict, optional): Dictionary of learning parameters, used to get vth.
        visualize_weights (bool): If True, plots the initial and final weight heatmaps. Defaults to True.
    """

    print("\nGenerating Plots...")

    # Plot all behavior for each layer in a single figure
    for i, layer in enumerate(lif_layer):
        # Dynamically select neurons to plot to avoid index errors
        num_neurons_in_layer = v_data[i].shape[1]
        
        fig = plt.figure(figsize=(12, 10))
        ax1 = plt.subplot(3, 1, 1)
        ax2 = plt.subplot(3, 1, 2)
        ax3 = plt.subplot(3, 1, 3)
        fig.suptitle(f'Layer {i} ({layer.name}) - Neuron Behavior', fontsize=16)

        # 1. Plot Voltage as a heatmap
        if not np.all(np.isfinite(v_data[i])):
            ax1.text(0.5, 0.5, "Voltage contains NaN or Inf", horizontalalignment="center", verticalalignment="center", transform=ax1.transAxes)
        else:
            im1 = ax1.imshow(v_data[i].T, aspect='auto', cmap='viridis')
            ax1.set_ylabel("Neuron Index")
            fig.colorbar(im1, ax=ax1, label="Voltage (v)")
        ax1.set_title("Voltage Heatmap")
        ax1.grid(False) # Grid is not useful for heatmaps

        # 2. Plot Current as a heatmap
        if not np.all(np.isfinite(u_data[i])):
            ax2.text(0.5, 0.5, "Current contains NaN or Inf", horizontalalignment="center", verticalalignment="center", transform=ax2.transAxes)
        else:
            im2 = ax2.imshow(u_data[i].T, aspect='auto', cmap='viridis')
            ax2.set_ylabel("Neuron Index")
            fig.colorbar(im2, ax=ax2, label="Current (u)")
        ax2.set_title("Current Heatmap")
        ax2.grid(False)

        # 3. Plot Spike Raster (already shows all neurons)
        spikes = s_data[i]
        if np.any(spikes):
            spike_times, spike_neurons = np.where(spikes)
            ax3.scatter(spike_times, spike_neurons, s=5)
        ax3.set_title("Spike Raster Plot")
        ax3.set_ylabel("Neuron Index")
        ax3.set_xlabel("Time Step")
        ax3.grid(True)

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.show()

    # Plot weight heatmaps for each plastic layer only if visualize_weights is True
    if visualize_weights:
        # Check if there are weights to plot
        if not initial_weights or not final_weights:
            print("Weight data not available, skipping weight plots.")
            return
            
        print("\nGenerating weight plots...")
        for i, layer in enumerate(plastic_layers):
            fig = plt.figure(figsize=(15, 6))
            ax1 = plt.subplot(1, 2, 1)
            ax2 = plt.subplot(1, 2, 2)
            fig.suptitle(f'Weight Analysis for {layer.name}', fontsize=16)

            # --- Heatmap of Initial Weights ---
            im1 = ax1.imshow(initial_weights[i], cmap='viridis', aspect='auto')
            ax1.set_title('Initial Weights')
            ax1.set_xlabel('Input Neuron Index')
            ax1.set_ylabel('Output Neuron Index')
            fig.colorbar(im1, ax=ax1)

            # --- Heatmap of Final Weights ---
            im2 = ax2.imshow(final_weights[i], cmap='viridis', aspect='auto')
            ax2.set_title('Final Weights')
            ax2.set_xlabel('Input Neuron Index')
            fig.colorbar(im2, ax=ax2)

            plt.show()