import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_test', type=int, default=10)
args = parser.parse_args()

print(args)