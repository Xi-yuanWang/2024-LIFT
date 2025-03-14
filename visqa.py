import argparse
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()
import json
with open(args.filename, "r") as f:
    a = json.load(f)
print(json.dumps(a["qa_pairs"], indent=2))
