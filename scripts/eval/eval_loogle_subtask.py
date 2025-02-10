import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
args = parser.parse_args()

with open(args.input, 'r') as f:
    data = [json.loads(l) for l in f]

results = {}
for sample in data:
    if 'qa_pairs' not in sample:
        continue
    for qa in sample['qa_pairs']:
        t = qa['type']
        if t not in results:
            results[t] = []
        results[t].append(qa['scores']['gpt4_score'])
for t in results:
    print(t, sum(results[t]) / len(results[t]))
