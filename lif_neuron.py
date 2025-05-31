# lif_neuron.py

import numpy as np

class LIFNeuron:
    def __init__(self, tau_m=20.0, v_rest=-65.0, v_reset=-65.0, v_thresh=-50.0,
                 r_m=1.0, dt=0.1, refractory=5):
        self.v = v_rest  # Membrane potential
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.tau_m = tau_m
        self.r_m = r_m
        self.dt = dt
        self.refractory_period = refractory
        self.refractory_timer = 0
        self.spike = False
        self.spike_times = []

    def reset(self):
        self.v = self.v_rest
        self.refractory_timer = 0
        self.spike = False
        self.spike_times = []

    def update(self, I_ext, t):
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            self.v = self.v_reset
            self.spike = False
        else:
            dv = self.dt * (-(self.v - self.v_rest) + self.r_m * I_ext) / self.tau_m
            self.v += dv
            self.spike = self.v >= self.v_thresh
            if self.spike:
                self.v = self.v_reset
                self.refractory_timer = self.refractory_period
                self.spike_times.append(t)
        return self.spike
