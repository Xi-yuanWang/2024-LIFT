import json
import numpy as np
import os

np.random.seed(0)
subtasks = ['gov_report', 'musique', 'narrativeqa', 'passage_retrieval_en', 'qmsum']
os.makedirs('datasets/longbench_tune/', exist_ok=True)
for subtask in subtasks:
    with open(f'datasets/longbench_sampling/{subtask}.jsonl', 'r') as f:
        data = [json.loads(l) for l in f]
    data = np.random.choice(data, 10, replace=False)
    with open(f'datasets/longbench_tune/{subtask}.jsonl', 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')
