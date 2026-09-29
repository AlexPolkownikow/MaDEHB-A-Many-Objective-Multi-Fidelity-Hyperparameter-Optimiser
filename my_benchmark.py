import numpy as np
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

class SyntheticMFBenchmark:
    def __init__(self):
        self.min_fidelity = 1
        self.max_fidelity = 50

        cs = ConfigurationSpace()
        cs.add_hyperparameter(UniformFloatHyperparameter("x", lower=-5, upper=5))
        self.cs = cs

    def get_configuration_space(self):
        return self.cs

    def query(self, config, fidelity):
        x = config["x"]
        noise = np.random.randn() * (1 / np.sqrt(fidelity))
        loss = (x - 2)**2 + noise
        return {"loss": loss}
