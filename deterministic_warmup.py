## we need to create an object for the deterministic warmup
## we can alternatively store the values of the betas as linspace

class DeterministicWarmup(object):

    def __init__(self, n_steps, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.increase = self.t_max / n_steps

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.increase

        self.t = self.t_max if t > self.t_max else t

        return self.t
