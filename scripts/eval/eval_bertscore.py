import argparse
import json
import tqdm
from bert_score import BERTScorer

parser = argparse.ArgumentParser()
parser.add_argument('-I', '--input', type=str, required=True)
args = parser.parse_args()

bert_scorer = BERTScorer(model_type='roberta-large', device='cuda')
bert_scores = []
with open(args.input, 'r') as f:
    for l in tqdm.tqdm(f.readlines()):
        data = json.loads(l)
        for qa_pair in data['qa_pairs']:
            reference = qa_pair['A']
            prediction = qa_pair['pred']
            bert_scores.append(bert_scorer.score([reference], [prediction])[-1].cpu().item())

print(sum(bert_scores) / len(bert_scores))
