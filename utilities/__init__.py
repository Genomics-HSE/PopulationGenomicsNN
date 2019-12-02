import msprime


LENGTH_NORMALIZE_CONST = 4


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
    Use as:
        >>> import utilities
        >>> dg = utilities.DataGenerator(0.1,0.1,[msprime.PopulationParametersChange(0,1)],10)
        >>> dg.run_simulation()
        >>> i = next(dg)
        >>> print(i)
            ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ([0], [0.20264850163110007]))

            (haplotypes after bite& , ([recombination points], [coalescent times]))

    """

    def __init__(self, recombination_rate: float,
                 mutation_rate: float,  demographic_events: list, num_replicates: int,
                 lengt: int = 10,  # 3*10**9
                 model: str = "hudson", random_seed: int = 42, sample_size: int = 2):
        self.sample_size = sample_size
        self.recombination_rate = recombination_rate
        self.mutation_rate = mutation_rate
        self.num_replicates = num_replicates
        self.demographic_events = demographic_events
        self.model = model
        self.len = lengt
        self.random_seed = random_seed

        self._data = None

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
        self._data = msprime.simulate(
            sample_size=self.sample_size,
            recombination_rate=self.recombination_rate,
            mutation_rate=self.mutation_rate,
            random_seed=self.random_seed,
            model=self.model,
            length=self.len,
            num_replicates=self.num_replicates,
            demographic_events=self.demographic_events
        )
        return self._data

    def __iter__(self):
        return self

    def __next__(self):
        """
        return haplotype, recombination points and coalescent time
        """
        if self._data is None:
            raise "Firstly you must run simulation"

        for replica in self._data:

            # TODO Защита от записи в один и тот же участок генома 
            haplotype = [0] * self.len

            for mutation in replica.mutations():
                point = round(mutation.position)
                haplotype[point] = 1

            recombination_points = []
            coal_times = []
            for tree in replica.trees():
                point = round(tree.get_interval()[0])
                if point not in recombination_points:
                    recombination_points.append(point)
                    coal_times.append(tree.total_branch_length /
                                      LENGTH_NORMALIZE_CONST)

            return (haplotype, (recombination_points, coal_times))
