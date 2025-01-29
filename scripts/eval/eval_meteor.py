"""
Compute the METEOR score for LooGLE.
"""
import argparse
import json
from nltk.translate.meteor_score import single_meteor_score


parser = argparse.ArgumentParser()
parser.add_argument('-I', '--input', type=str, required=True)
args = parser.parse_args()

meteor_scores = []
with open(args.input, 'r') as f:
    for l in f:
        data = json.loads(l)
        for qa_pair in data['qa_pairs']:
            reference = qa_pair['A'].split()
            prediction = qa_pair['pred'].split()
            meteor_scores.append(single_meteor_score(reference, prediction))

print(sum(meteor_scores) / len(meteor_scores))
