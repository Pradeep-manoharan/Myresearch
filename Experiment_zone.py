# Let's learning how to implement STDP learning rule in Lava
# Let's this is tag to "fixed_pt" and "floating_pt"
import numpy as np

SECECT_TAG = "floating_pt"
num_neuron = 1

if SECECT_TAG == "fixed_pf":
    dv = 4095
    du = 4095

if SECECT_TAG == "floating_pt":
    dv = 0
    du = 0

vth = 240

# Define the network and input
num_neuron = 1
shape_lif = (num_neuron,)
shape_connection = (num_neuron,num_neuron)

wgt_input = np.eye(num_neuron) * 250
wgt_plast_conn = np.full(shape_connection,50)
num_step = 200
time = list(range(1,num_step+1))
spike_prob = 0.03

# Generate the spike input
np.random.seed(23)
# Pre-synaptic
spike_raster_pre = np.zeros((num_neuron,num_step))
np.place(spike_raster_pre,np.random.rand(num_neuron,num_step)< spike_prob, 1)

# Post_synaptic
spike_raster_post = np.zeros((num_neuron,num_step))
np.place(spike_raster_post,np.random.rand(num_neuron,num_step)< spike_prob, 1)


# Define the learning rule
from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi
stdp = STDPLoihi(learning_rate = 1,A_plus = 1,
                  A_minus = -1,tau_plus = 10,
                  tau_minus = 10, t_epoch = 4,)

# Build the network
# Spike_Generator --> Dense --> LIF(pre) --> LearningDense --> LIF(post)
from lava.proc.lif.process import LIF
from lava.proc.io.source import RingBuffer
from lava.proc.dense.process import LearningDense, Dense

# Create input spike
pattern_pre = RingBuffer(data = spike_raster_pre.astype(int))
pattern_post = RingBuffer(data = spike_raster_post.astype(int))

# Create input connectivity
conn_inp_pre = Dense(weights= wgt_input)
conn_inp_post = Dense(weights= wgt_input)

# Create pre-synaptic 
lif_pre = LIF(u = 0,
              v = 0,
              du = du,
              dv = dv,
              bias_mant = 0,
              bias_exp = 0,
              vth = vth,
              shape = shape_lif,
              name = "lif_pre")

# Create plastic connection
plast_conn = LearningDense(weights = wgt_plast_conn,learning_rule = stdp,name = "plastic dense")

# Create post-synaptics
lif_post = LIF(v = 0,
               u = 0,
               du = 0,
               dv = 0,
               bias_mant = 0,
               bias_exp = 0,
               vth = vth,
               shape = shape_lif,
               name = "lif_post")

# Connect the network
pattern_pre.s_out.connect(conn_inp_pre.s_in)
conn_inp_pre.a_out.connect(lif_pre.a_in)

pattern_post.s_out.connect(conn_inp_post.s_in)
conn_inp_post.a_out.connect(lif_post.a_in)

lif_pre.s_out.connect(plast_conn.s_in)
plast_conn.a_out.connect(lif_post.a_in)

# Connect back-progating action potential
lif_post.s_out.connect(plast_conn.s_in_bap)


from lava.proc.monitor.process import Monitor
# Create Monitor
mom_pre_trace = Monitor()
mom_post_trace = Monitor()
mom_pre_spike = Monitor()
mom_post_spike = Monitor()
mom_weight = Monitor()

# Connect Monitor
mom_pre_trace.probe(plast_conn.x1,num_step)
mom_post_trace.probe(plast_conn.y1,num_step)
mom_pre_spike.probe(lif_pre.s_out,num_step)
mom_post_spike.probe(lif_post.s_out, num_step)
mom_weight.probe(plast_conn.weights, num_step)

# Let's run the simulation

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg

if __name__ == "__main__":
    # Running
    pattern_pre.run(condition = RunSteps(num_steps = num_step), 
                    run_cfg = Loihi2SimCfg(select_tag = SECECT_TAG))
    
    # Get dat from monitors
    pre_trace = mom_pre_trace.get_data()["plastic dense"]["x1"]
    post_trace = mom_post_trace.get_data()["plastic dense"]["y1"]
    pre_trace = mom_pre_spike.get_data()["lif_pre"]["s_out"]
    post_trace = mom_post_spike.get_data()["lif_post"]["s_out"]
    weights = mom_weight.get_data()["plastic dense"]["weights"][:,:,0]
    #print(weights.shape)
    
    # Stopping 
    pattern_pre.stop()

    import matplotlib.pyplot as plt     
    # plot spike trainss
    def plot_spikes(spikes,legend,colors):
        offsets = list(range(1,len(spikes) +1))
        plt.figure(figsize =(10,3))

        spikes_plot = plt.eventplot(positions = spikes,
                                    lineoffsets = offsets,
                                    linelength = 0.9,
                                    color = colors)
        plt.title("Spike arival")
        plt.xlabel("Time Steps")
        plt.ylabel("Neuron")
        plt.yticks(ticks= offsets,labels = legend)
        plt.show()    

    # plot spikes
    # plot_spikes(spikes = [np.where(pre_trace[:,0])[0],np.where(post_trace[:,0])[0]], legend = ['Post','pre'],
                #colors=['#370665', '#f14a16'])


    # Plotting trace dynamics
    def plot_time(time,time_series,ylabel,title,):
        plt.figure(figsize = (10,1))
        plt.step(time,time_series)
        plt.title(title)
        plt.xlabel("Time steps")
        plt.ylabel(ylabel)
        plt.show()

    # Plotting pre trace dynamics
    # plot_time(time = time,time_series = pre_trace,ylabel = "Trace Value",title = "Pre Trace")
    # plot_time(time = time, time_series= post_trace, ylabel = "Trace Value", title = "Post Trace")
    # plot_time(time = time, time_series = weights, ylabel = "Weight Value",title = "Weight dynamic")

    # w_diff = np.zeros(weights.shape)
    # print(weights[0:])
    # print(np.diff(weights))
    # w_diff[1:] = np.diff(weights)
    # print(w_diff)








    