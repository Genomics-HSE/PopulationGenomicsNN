import argparse
import numpy as np
import utilities


parser = argparse.ArgumentParser()
parser.add_argument('-Ne', type=float, default=1.0, help='Ask VL')
parser.add_argument('-recombination_rate', type=float,
                    default=1.6*10e-9, help='')
parser.add_argument('-mutation_rate', type=float, default=1.25*10e-8, help='')
parser.add_argument('-num_replicates', type=int, default=int(1), help='')
parser.add_argument('-lengt', type=int, default=int(10), help='')
parser.add_argument('-random_seed', default=21092020, help='')
parser.add_argument('-POPULATION', default=5000, help='')
parser.add_argument('-is_experement', default=True, help='')
parser.add_argument('-num_intervals', default=int(20), help='')
parser.add_argument('-targets_file_name', default="label.npy", help='')
parser.add_argument('-features_file_name', default="feature.npy", help='')

args = parser.parse_args()
"""
args = parser.parse_args()
random_seed = 21092020
RHO_HUMAN = 1.6*10e-9
MU_HUMAN = 1.25*10e-8
POPULATION = 5000
num_replicates = 1
lengt = 3000000000
is_experement = True
num_intervals = 20

targets_file_name = "label.npy"
features_file_name = "feature.npy"
"""

dg = utilities.DataGenerator(
    recombination_rate=args.recombination_rate,
    mutation_rate=args.mutation_rate,
    demographic_events=utilities.generate_demographic_events(args.POPULATION),
    num_replicates=args.num_replicates,
    lengt=args.lengt,
    random_seed=args.random_seed,
    N=args.num_intervals,
    is_experement=args.is_experement
)


dg.run_simulation()

data = next(dg)

haplotype, d_times, recombination_points, prioty_distribution, intervals_starts = data

with open(args.features_file_name, 'wb') as f:
    np.save(f, haplotype)

with open(args.targets_file_name, 'wb') as f:
    np.save(f, d_times)
