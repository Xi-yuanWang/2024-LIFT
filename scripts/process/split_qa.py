import json
import argparse
import random
import copy

def main(args):
    a = list(range(5))

    with open(args.input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    for d in data:
        d['qa_pairs'] = eval(d['qa_pairs'])
    
    data = sorted(data, key=lambda d: len(d['qa_pairs']), reverse=True)
    data = [d for d in data if len(d['qa_pairs']) > 10]

    for d in data:
        random.shuffle(d['qa_pairs'])
        d['train_qa_pairs'] = d['qa_pairs'][:10]
        d['test_qa_pairs'] = d['qa_pairs'][10:]
        # d['train_qa_pairs'] = d['qa_pairs'][-9:]
        # d['test_qa_pairs'] = d['qa_pairs'][:-9]

        d.pop('qa_pairs')
    
    with open(args.output_file, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parsed_args = parser.parse_args()

    main(parsed_args)
