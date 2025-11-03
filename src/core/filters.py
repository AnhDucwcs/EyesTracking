import numpy as np
import time
import math


class EMAFilter:
    def __init__(self, alpha=0.8, init=None):
        self.alpha = alpha
        self.state = None if init is None else np.array(init, dtype=np.float32)


    def update(self, measurement):
        m = np.array(measurement, dtype=np.float32)
        if self.state is None:
            self.state = m
        else:
            self.state = self.alpha * self.state + (1.0-self.alpha) * m
        return self.state


   
class OneEuroWrapper:
    def __init__(self, freq=60, min_cutoff=0.1, beta=1.0, deriv_cutoff=1.0, init_val=(0.5, 0.5)):
        self.freq = freq
        config = {
            'min_cutoff': min_cutoff,
            'beta': beta,
            'd_cutoff': deriv_cutoff
        }
        x0, y0 = init_val
        t0 = time.time()
        self.filter_x = OneEuroFilter(t0, x0=x0, **config)
        self.filter_y = OneEuroFilter(t0, x0=y0, **config)

    def update(self, measurement):
        current_time = time.time()
        raw_x, raw_y = measurement
        smooth_x = self.filter_x(current_time, raw_x)
        smooth_y = self.filter_y(current_time, raw_y)
        
        return (smooth_x, smooth_y)
    
    
    
def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev
        if t_e < 1e-6:
            return self.x_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat