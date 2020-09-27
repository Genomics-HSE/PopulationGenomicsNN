import numpy as np
import torch

import msprime
from math import (exp, log)

RHO_HUMAN = 1.6*10e-9
MU_HUMAN = 1.25*10e-8
RHO_LIMIT = (log(RHO_HUMAN)-100, log(RHO_HUMAN)+100)
MU_LIMIT = (log(MU_HUMAN)-100, log(MU_HUMAN)+100)

LENGTH_NORMALIZE_CONST = 4
ZIPPED = False
NUMBER_OF_EVENTS_LIMITS = (1, 20)
MAX_T_LIMITS = (0.01, 30)
LAMBDA_EXP = 1.0
POPULATION_LIMITS = (250, 100000)
POPULATION = 5000

N = 20

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def generate_demographic_events(popilation: int = POPULATION) -> list:
    """
    Generate demographic events.
    1) We generate number of events
    2) We choise time when events happens
    3) We choise how population has changed

    For more information learn msprime documentation


    Must return list of msprime.PopulationParametersChange objects
    https://msprime.readthedocs.io/en/stable/api.html#demographic-events
    """
    number_of_events = np.random.randint(
        low=NUMBER_OF_EVENTS_LIMITS[0], high=NUMBER_OF_EVENTS_LIMITS[1])
    max_t = np.random.uniform(low=MAX_T_LIMITS[0], high=MAX_T_LIMITS[1])

    times = sorted(np.random.exponential(LAMBDA_EXP, size=number_of_events))

    alpha = 1.0
    beta = np.log(max_t + 1)/times[-1]

    def to_exp_time(time: float) -> float:
        # time -> exponentional time
        return alpha*(np.exp(beta*time) - 1)

    exp_times = [to_exp_time(t) for t in times]
    # population_sizes = np.random.randint(
    #    low=POPULATION_LIMITS[0], high=POPULATION_LIMITS[1], size=number_of_events)

    population_sizes = np.random.beta(
        a=2, b=5, size=number_of_events)*popilation

    # init_population = np.random.randint(
    #    low=POPULATION_LIMITS[0], high=POPULATION_LIMITS[1])

    init_population = int(np.random.beta(a=2, b=5)*popilation)

    events = [msprime.PopulationParametersChange(
        0, initial_size=init_population, growth_rate=0)]

    for t, s in zip(exp_times, population_sizes):
        events.append(
            msprime.PopulationParametersChange(t, int(s), growth_rate=0)
        )
    return events


def give_rho() -> float:
    return exp(np.random.uniform(RHO_LIMIT[0], RHO_LIMIT[1]))


def give_mu() -> float:
    return exp(np.random.uniform(MU_LIMIT[0], MU_LIMIT[1]))


def give_random_rho(base=RHO_HUMAN) -> float:
    return np.random.uniform(0.0001, 100, 1)[0]*base/10


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
                 model: str = "hudson", random_seed: int = 42, sample_size: int = 2, N: int = N, is_experement: bool = False):
        self.sample_size = sample_size
        self.recombination_rate = recombination_rate
        self.mutation_rate = mutation_rate
        self.num_replicates = num_replicates
        self.demographic_events = demographic_events
        self.model = model
        self.len = lengt
        self.random_seed = random_seed
        self.is_experement = is_experement
        self.N = int(N)

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

        # for j, replica in enumerate(self._data):

        try:
            replica = next(self._data)
        except StopIteration:
            raise StopIteration

        # TODO Защита от записи в один и тот же участок генома 
        haplotype = [0] * self.len

        for mutation in replica.mutations():
            point = round(mutation.position)
            if point < self.len:
                haplotype[point] = 1
            else:
                haplotype[point - 1] = 1

        recombination_points = []
        coal_times = []
        for tree in replica.trees():
            point = round(tree.get_interval()[0])
            if point not in recombination_points:
                recombination_points.append(point)
                coal_times.append(tree.total_branch_length /
                                  LENGTH_NORMALIZE_CONST)

        # return (haplotype, (recombination_points, coal_times))

        # haplotype = "".join([str(h) for h in haplotype])
        times = [.0] * len(haplotype)
        j_point = 0
        j_time = -1
        time = None
        for i, _ in enumerate(times):
            if j_point < len(recombination_points):
                if i == recombination_points[j_point]:
                    j_point += 1
                    j_time += 1
                    time = coal_times[j_time]
            times[i] = time

        min_t = min(times)
        max_t = max(times)

        print(self.N)
        a = (-np.log(max_t) + self.N*np.log(min_t))/(self.N-1)
        from pprint import pprint
        # pprint(times)
        print(min_t)
        print(max_t)
        B = (-np.log(min_t) + np.log(max_t))/(self.N-1) + 10**(-10)

        def to_T(time):
            return round((np.log(time)-a)/B)

        step_of_discratization = max(times)/self.N

        def discretization(t):
            return min(int(t/step_of_discratization) + 1, self.N)

        #d_times = [discretization(t) for t in times]
        #from pprint import pprint
        # pprint(times)
        #print(f"a {a}, B {B}")
        d_times = [to_T(t) for t in times]

        if self.is_experement:
            print("Experimente mode on")
            prioty_distribution = [0.0 for i in range(self.N+1)]
            for t in d_times:
                prioty_distribution[t] += 1
            prioty_distribution = [p/sum(prioty_distribution)
                                   for p in prioty_distribution]

            intervals_starts = [np.e**(B*i+a) for i in range(self.N)]
            return np.array(haplotype), d_times, recombination_points, prioty_distribution, intervals_starts
        else:
            return (np.array(haplotype), d_times, recombination_points)
        # return (np.array(haplotype), times, recombination_points)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, z):
        super(MyDataset, self).__init__()
        assert x.shape[0] == y.shape[0]  # assuming shape[0] = dataset size
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index]


def make_dataset(args):
    """Create data generator"""

    events = generate_demographic_events()

    dg = DataGenerator(recombination_rate=args.rho,
                       mutation_rate=args.mu,
                       demographic_events=events,
                       num_replicates=args.num_repl, lengt=args.l)
    dg.run_simulation()

    """Create datasets"""

    number_train_examples = int(args.num_repl*args.ratio_train_examples)

    trX, trY, trZ = [], [], []
    for _ in range(number_train_examples):
        example = next(dg)
        trX.append(example[0])
        trY.append(example[1])
        trZ.append(example[2])

    teX, teY, teZ = [], [], []
    for example in dg:
        teX.append(example[0])
        teY.append(example[1])
        teZ.append(example[2])

    del dg

    input = torch.from_numpy(np.array(trX, dtype=np.float_))  # .to(device)
    target = torch.from_numpy(np.array(trY))  # .to(device)
    extra_target = torch.from_numpy(np.array(trZ))
    test_input = torch.from_numpy(
        np.array(teX, dtype=np.float_))  # .to(device)
    test_target = torch.from_numpy(np.array(teY))  # .to(device)
    test_extra_target = torch.from_numpy(np.array(teZ))

    del trX, trY, teX, teY, trZ, teZ

    return MyDataset(input, target, extra_target), MyDataset(test_input, test_target, test_extra_target)
