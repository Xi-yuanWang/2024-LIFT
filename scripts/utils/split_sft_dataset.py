"""
Construct SFT dataset from QuALITY.
"""
import json
import tqdm

write_f = open('data/sft_train.jsonl', 'w')
with open('datasets/QuALITY/QuALITY.v1.0.1.htmlstripped.train', 'r') as f:
    for sample in tqdm.tqdm(f.readlines(), desc="QuALITY"):
        sample = json.loads(sample)
        write_f.write(json.dumps({
            'article_id': sample['article_id'],  # used to identify datapoints
            'title': sample['title'],
            'input': sample['article']
        }) + '\n')
write_f.close()
