import msprime


def generate_demographic_events(random_seed: int = 42) -> list:
    """
    Generate demographic events.
    1) We generate number of events
    2) We choise time when events happens
    3) We choise how population has changed

    For more information learn msprime documentation
    """
    pass


class DataGenerator():
    """
    """

    def __init__(self, sample_size: int, recombination_rate: float,
                 mutation_rate: float, model: str, demographic_events: list,
                 num_replicates: int, random_seed: int):
        self.sample_size = sample_size
        self.recombination_rate = recombination_rate
        self.mutation_rate = mutation_rate
        self.num_replicates = num_replicates
        self.demographic_events = demographic_events
        self.model = model
        self.random_seed = random_seed

    def __str__(self):
        return """DataGenerator class:\n sample_size = {}\n
         recombination_rate = {}\n mutation_rate = {}\n
         model = {}\n num_replicates = {}\n
          demographic_events = {}\n
          random_seed = {}\n""".format(
            self.sample_size,
            self.recombination_rate,
            self.mutation_rate,
            self.model,
            self.num_replicates,
            self.demographic_events,
            self.random_seed
        )

    def run_simulation(self):
        """
        return generator(tskit.TreeSequence)
        function run the simulation with given parametrs
        """
        return msprime.simulate(
            sample_size=self.sample_size,
            recombination_rate=self.recombination_rate,
            mutation_rate=self.mutation_rate,
            random_seed=self.random_seed,
            model=self.model,
            num_replicates=self.num_replicates,
            demographic_events=self.demographic_events
        )

    def __call__(self):
        pass
