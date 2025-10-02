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

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

# Lava imports
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.monitor.process import Monitor
from lava.proc.lif import models


from module import Encoder, OutputProcess, build_network, WeightNormalizer,plot_neuron_behaviour, evaluation
  
if __name__ == "__main__":
    # General Parameters
    class_names = [str(i) for i in range(10)]

    """Leaky-Integrate-and-Fire (LIF) neural Process.

    LIF dynamics abstracts to:
    u[t] = u[t-1] * (1-du) + a_in         # neuron current
    v[t] = v[t-1] * (1-dv) + u[t] + bias  # neuron voltage
    s_out = v[t] > vth                    # spike if threshold is exceeded
    v[t] = 0                              # reset at spike 
    """
    num_image = 100
    step_per_image = 5
    norm_interval = step_per_image
    num_step = num_image * step_per_image
    neuron_count_visual = int(784)
    time_count_visual = num_step+1
    layer_sizes = [784,128,10]
    GET_WEIGHT_DATA = False # Set to False to skip weight collection and plotting
    USE_WEIGHT_NORMALIZER = False # Set to False to disable weight normalization
    USE_LATERAL_INHIBITION = False # Set to False to disable the lateral inhibition
  

    learning_params = {"dv": [0.1,0.2,0.5], "du" :[0.8,0.8,0.9] , "vth" : [10,11,12], "learning_rate" : 1,
                       "A_plus" : 0.01, "A_minus" : 0.01, "tau_plus" : 5,"tau_minus" : 5, "t_epoch" : 5, "bias_mant" : 0}
        
    # Processes
    encoder = Encoder(n_img = num_image, num_step_per_image = step_per_image,vth = 1)
    decoder = OutputProcess(num_step=step_per_image,n_img = num_image)


    # Build the network layers using our new function
    lif_layers, plastic_layers = build_network(layer_sizes,
                                               dv_vals=learning_params["dv"],
                                               du_vals=learning_params["du"],
                                               vth=learning_params["vth"],
                                               bias_mant=learning_params["bias_mant"],
                                               learning_rate=learning_params["learning_rate"],
                                               a_plus=learning_params["A_plus"],
                                               a_minus=learning_params["A_minus"],
                                               tau_plus=learning_params["tau_plus"],
                                               tau_minus=learning_params["tau_minus"],
                                               t_epoch=learning_params["t_epoch"],
                                               use_lateral_inhibition = USE_LATERAL_INHIBITION)

    # Lets Make the connection
    encoder.outport.connect(lif_layers[0].a_in)
    

    #Create and attach a normalizer for each plastic layer
    if USE_WEIGHT_NORMALIZER:
        print("Weight Normalization is ENABLED")
        normalize_proc = []
        if plastic_layers:
            # Connect the encoder label out to the *first* normalizer's trigger
            wn = WeightNormalizer(plastic_layers[0].weights, norm_interval, name=plastic_layers[0].name)
            normalize_proc.append(wn)
            encoder.label_out.connect(wn.trigger_in) # Encoder -> WN[0]
            print(f"Attaching weightNormalizer to {plastic_layers[0].name}")

            # Chain the rest of the normalizers by fanning out from the first trigger input
            for i in range(1, len(plastic_layers)):
                next_wn = WeightNormalizer(plastic_layers[i].weights, norm_interval, name=plastic_layers[i].name)
                normalize_proc.append(next_wn)
                wn.trigger_in.connect(next_wn.trigger_in) # WN[0] -> WN[i]
                print(f"Attaching weightNormalizer to {plastic_layers[i].name}")
    else:
        print("Weight Normalization is DISABLED")
        encoder.label_out.connect(decoder.label_in)

    # Connect the LIF layers and plastic layers

    for i in range(len(plastic_layers)):

        lif_layers[i].s_out.connect(plastic_layers[i].s_in)
        print(f'Connection is made between {lif_layers[i].name} and {plastic_layers[i].name}')
        plastic_layers[i].a_out.connect(lif_layers[i+1].a_in)
        print(f'Connection is made betweeen {plastic_layers[i].name} and {lif_layers[i+1].name}')
        
        if hasattr(plastic_layers[i],"s_in_bap"):
            lif_layers[i+1].s_out.connect(plastic_layers[i].s_in_bap)
            print(f'The BAP connection also made between {lif_layers[i+1].name} and {plastic_layers[i].name}')
        print()

    lif_layers[-1].s_out.connect(decoder.in_port)
    print("Connection simulation")


    # Let's Create the monitor
    """
    Here, we gonna visualize the spike out of the encoder and each LIF layers of v, u and 
    s_out and weight change of dense layer
    """

    # Let's Visulize the spike out from decoder
    mon_s_out_encoder = Monitor(); mon_s_out_encoder.probe(encoder.outport,num_step)

    # Let's Visualize the V,U, and S_out of lif layer
    mon_lif_v, mon_lif_u, mon_lif_s_out = [],[],[]
    for mon_lif in lif_layers:
        lif_mon_v = Monitor(); lif_mon_v.probe(mon_lif.v,num_step); mon_lif_v.append(lif_mon_v)
        lif_mon_u = Monitor(); lif_mon_u.probe(mon_lif.u,num_step); mon_lif_u.append(lif_mon_u)
        lif_mon_s_out = Monitor(); lif_mon_s_out.probe(mon_lif.s_out,num_step); mon_lif_s_out.append(lif_mon_s_out)

    # Initialize weight lists
    initial_weights, final_weights = [], []

    # Conditionally store the initial weights before the run
    if GET_WEIGHT_DATA:
        print("Collecting Initial Weights...")
        initial_weights = [np.copy(pl.weights.get()) for pl in plastic_layers]

    # Running:
    try:
        encoder.run(condition=RunSteps(num_steps=num_step),
                    run_cfg=Loihi2SimCfg(select_tag="floating_pt"))

        # Get the monitor data
        print("Creating monitor data")
        v_data = [m.get_data()[lif.name]["v"]
                  for m, lif in zip(mon_lif_v, lif_layers)]
        print("v_data created")
        u_data = [m.get_data()[lif.name]["u"]
                  for m, lif in zip(mon_lif_u, lif_layers)]
        print("u_data created")
        s_data = [m.get_data()[lif.name]["s_out"]
                  for m, lif in zip(mon_lif_s_out, lif_layers)]
        print("s_data created")

        # Conditionally get the final weights after the run
        if GET_WEIGHT_DATA:
            print("Collecting Final Weights...")
            final_weights = [np.copy(pl.weights.get()) for pl in plastic_layers]
            print("w_data created")

        ground_truth_label = decoder.ground_truth_labels.get().astype(np.int32)
        prediction_label = decoder.predicted_labels.get().astype(np.int32)
        
        # Enhanced evaluation
        report, cm, accuracy = evaluation.evaluate_performance(
            ground_truth=ground_truth_label,
            predictions=prediction_label,
            class_names=class_names
        )

        for i, layer in enumerate(lif_layers):
            print(f"Layer {i} ({layer.name}): Min v = {np.min(v_data[i])}, Max v = {np.max(v_data[i])} and {v_data[i].shape}")
            print(f"Layer {i} ({layer.name}): Min u = {np.min(u_data[i])}, Max u = {np.max(u_data[i])} and {v_data[i].shape}")
            print(f"Layer {i} ({layer.name}): Min s = {np.min(s_data[i])}, Max s = {np.max(s_data[i])} and {v_data[i].shape}")
    
    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        # Stop the simulation
        encoder.stop()
        # Plot the results
        plot_neuron_behaviour(v_data, neuron_count_visual, time_count_visual,
                              u_data=u_data, s_data=s_data,
                              initial_weights=initial_weights, final_weights=final_weights,
                              lif_layer=lif_layers, plastic_layers=plastic_layers, learning_params=learning_params,
                              visualize_weights=GET_WEIGHT_DATA)

        # Save the results, checking if variables exist before using them
        results = {
            "v_data": v_data if 'v_data' in locals() else [],
            "u_data": u_data if 'u_data' in locals() else [],
            "s_data": s_data if 's_data' in locals() else [],
            "initial_weights": initial_weights,
            "final_weights": final_weights,
            "ground_truth_label": ground_truth_label if 'ground_truth_label' in locals() else None,
            "prediction_label": prediction_label if 'prediction_label' in locals() else None,
            "accuracy": accuracy if 'accuracy' in locals() else None,
            "report": report if 'report' in locals() else None,
            "confusion_matrix": cm if 'cm' in locals() else None
        }
        with open("sim_results.pkl", "wb") as f:
            pickle.dump(results, f)
        print("\nResults saved to sim_results.pkl")