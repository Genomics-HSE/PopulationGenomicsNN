import msprime

# see the tutorial in https://msprime.readthedocs.io/en/stable/tutorial.html


def full_arg_example():
    ts = msprime.simulate(sample_size=5, recombination_rate=0.1,
                          record_full_arg=True, random_seed=42)
    print(ts.tables.nodes)
    print()
    for tree in ts.trees():
        print("interval:", tree.interval)
        print(tree.draw(format="unicode"))


def create_model():
    """
    define msprime model that will be used in simulation
    See:
    https://msprime.readthedocs.io/en/stable/api.html#sec-api-simulation-models
    ? better to use arg
    """
    return "smc_prime"


def run_simulatin(sample_size: int, recombination_rate: float,
                  mutation_rate: float, model, random_seed: int = 42):
    """
    return tskit.TreeSequence
    function run the simulation with given parametrs
    """
    return msprime.simulate(
        sample_size=sample_size,
        recombination_rate=recombination_rate,
        mutation_rate=mutation_rate,
        random_seed=random_seed,
        model=model
    )


full_arg_example()
