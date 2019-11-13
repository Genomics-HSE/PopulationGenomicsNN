from .generator import *


class DataGenerator():
    """
    """

    def __init__(self, sample_size: int, recombination_rate: float,
                 mutation_rate: float, model: str, random_seed: int = 42):
        self.sample_size = sample_size
        self.recombination_rate = recombination_rate
        self.mutation_rate = mutation_rate
        self.model = model
        self.random_seed = random_seed

    def __str__(self):
        return """DataGenerator class:\n sample_size = {}\n
         recombination_rate = {}\n mutation_rate = {}\n
         model = {}\n random_seed={}""".format(
            self.sample_size,
            self.recombination_rate,
            self.mutation_rate,
            self.model,
            self.random_seed
        )

    def __call__(self):
        pass

    def generate(self):
        pass
