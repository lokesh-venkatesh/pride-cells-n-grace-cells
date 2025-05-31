import numpy as np
import matplotlib.pyplot as plt

class LIFNeuron:
    def __init__(self, tau=20.0, v_rest=0.0, v_reset=0.0, v_threshold=1.0, r=1.0):
        self.tau = tau
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.r = r
        self.v = self.v_rest
        self.spike = False

    def reset(self):
        self.v = self.v_reset
        self.spike = False

    def update(self, I_ext, dt):
        dv = (-(self.v - self.v_rest) + self.r * I_ext) / self.tau
        self.v += dv * dt
        self.spike = self.v >= self.v_threshold
        if self.spike:
            self.reset()
        return self.spike

class Synapse:
    def __init__(self, pre_neurons, post_neurons, learning_rate=0.01, tau_pre=20.0, tau_post=20.0):
        self.pre = pre_neurons
        self.post = post_neurons
        self.weights = np.random.normal(0.5, 0.1, (len(post_neurons), len(pre_neurons)))
        self.learning_rate = learning_rate
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.trace_pre = np.zeros(len(pre_neurons))
        self.trace_post = np.zeros(len(post_neurons))

    def propagate(self):
        pre_spikes = np.array([n.spike for n in self.pre], dtype=np.float32)
        I_post = self.weights @ pre_spikes
        return I_post

    def update_traces(self, dt):
        self.trace_pre *= np.exp(-dt / self.tau_pre)
        self.trace_post *= np.exp(-dt / self.tau_post)
        for i, n in enumerate(self.pre):
            if n.spike:
                self.trace_pre[i] += 1.0
        for j, n in enumerate(self.post):
            if n.spike:
                self.trace_post[j] += 1.0

    def apply_r_stdp(self, reward):
        for i, post_n in enumerate(self.post):
            for j, pre_n in enumerate(self.pre):
                dw = self.learning_rate * reward * (self.trace_pre[j] * post_n.spike - self.trace_post[i] * pre_n.spike)
                self.weights[i, j] += dw
        np.clip(self.weights, 0.0, 1.0, out=self.weights)

class NeuronGroup:
    def __init__(self, size, neuron_type='excitatory', **neuron_params):
        self.neurons = [LIFNeuron(**neuron_params) for _ in range(size)]
        self.size = size
        self.type = neuron_type

    def reset(self):
        for neuron in self.neurons:
            neuron.reset()

    def update(self, input_current, dt):
        spikes = [neuron.update(I, dt) for neuron, I in zip(self.neurons, input_current)]
        return spikes

    def get_spikes(self):
        return np.array([n.spike for n in self.neurons], dtype=np.float32)

    def get_voltages(self):
        return np.array([n.v for n in self.neurons])

class EINetwork:
    def __init__(self, num_exc=4, num_inh_groups=1, num_inh_per_group=1):
        self.E = NeuronGroup(num_exc, neuron_type='excitatory')
        self.I_groups = [NeuronGroup(num_inh_per_group, neuron_type='inhibitory',
                                    v_threshold=0.5, tau=10.0)  # easier to spike inhibitory neurons
                         for _ in range(num_inh_groups)]

        self.EE_syn = Synapse(self.E.neurons, self.E.neurons)
        self.EI_syn = [Synapse(self.E.neurons, I_group.neurons) for I_group in self.I_groups]
        self.IE_syn = [Synapse(I_group.neurons, self.E.neurons) for I_group in self.I_groups]

        self.history = {
            'E_voltages': [],
            'E_spikes': [],
            'I_voltages': [[] for _ in range(num_inh_groups)],
            'I_spikes': [[] for _ in range(num_inh_groups)]
        }

    def step(self, ext_input, reward=0.0, dt=1.0):
        # Feedforward from external input + inhibition from I neurons
        I_exc = ext_input + self.IE_total_inhibition()
        spikes_E = self.E.update(I_exc, dt)

        # Update inhibitory groups with EI synaptic input + small baseline input
        baseline_I_ext = 0.05  # small external drive to inhibitory neurons

        for i, I_group in enumerate(self.I_groups):
            I_inh = self.EI_syn[i].propagate()
            I_total_inh = I_inh + baseline_I_ext
            spikes_I = I_group.update(I_total_inh, dt)

            # Debug prints for inhibitory group inputs and spikes
            if (len(self.history['E_voltages']) % 50) == 0:  # print every 50 steps for less clutter
                print(f"Step {len(self.history['E_voltages'])}: I-group {i} input current (mean): {np.mean(I_total_inh):.3f}, spikes: {sum(spikes_I)}")

        # Apply R-STDP on EE synapses only for now
        self.EE_syn.update_traces(dt)
        self.EE_syn.apply_r_stdp(reward)

        # Log history
        self.history['E_voltages'].append(self.E.get_voltages())
        self.history['E_spikes'].append(self.E.get_spikes())
        for i, I_group in enumerate(self.I_groups):
            self.history['I_voltages'][i].append(I_group.get_voltages())
            self.history['I_spikes'][i].append(I_group.get_spikes())

    def IE_total_inhibition(self):
        I_total = np.zeros(self.E.size)
        for syn in self.IE_syn:
            I_total += syn.propagate()
        return -I_total

    def plot_activity(self):
        fig, axs = plt.subplots(2 + len(self.I_groups), 1, figsize=(10, 6))
        E_spikes = np.array(self.history['E_spikes'])
        axs[0].imshow(E_spikes.T, aspect='auto', cmap='binary')
        axs[0].set_title('Excitatory Spikes')

        E_volt = np.array(self.history['E_voltages'])
        axs[1].plot(E_volt)
        axs[1].set_title('Excitatory Voltages')

        for i in range(len(self.I_groups)):
            I_spikes = np.array(self.history['I_spikes'][i])
            axs[2+i].imshow(I_spikes.T, aspect='auto', cmap='binary')
            axs[2+i].set_title(f'Inhibitory Group {i+1} Spikes')

        plt.tight_layout()
        plt.show()

    def get_E_firing_rates(self, window=10):
        """
        Returns average firing rate of each E neuron over last 'window' timesteps.
        """
        spikes_history = np.array(self.history['E_spikes'][-window:])  # shape: (window, num_E)
        if spikes_history.size == 0:
            return np.zeros(self.E.size)
        return np.mean(spikes_history, axis=0)
