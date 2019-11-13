import argparse

parser = argparse.ArgumentParser(
    description='Process parametrs to run simulation')

parser.add_argument('-rs', '--random-seed', type=int, default=42)
parser.add_argument('-sz', '--sample-size', type=int, required=True)
parser.add_argument('-rr', '--recombination-rate', type=float, required=True)
parser.add_argument('-mr', '--mutation-rate', type=float, required=True)
parser.add_argument('-m', '--model', type=str, required=True)

args = parser.parse_args()
