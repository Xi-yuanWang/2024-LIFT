import argparse
import os
import subprocess
import json

MP_INPUT_DIR = 'mp/input'
MP_OUTPUT_DIR = 'mp/output'

def main(args):
    os.makedirs(MP_INPUT_DIR, exist_ok=True)
    os.makedirs(MP_OUTPUT_DIR, exist_ok=True)

    with open(args.input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    if args.num_test:
        data = data[:args.num_test]
    else:
        args.num_test = len(data)

    # prepare output filename
    mp_output_files = [os.path.join(MP_OUTPUT_DIR, f'{os.path.basename(args.output_file)}_num_{args.num_test}_{i}_in_{args.num_process}') for i in range(args.num_process)]
    print('mp_output_files: ', mp_output_files)

    # get output res
    with open(args.original_output_file, 'r') as f:
        output = [json.loads(line) for line in f]

    # distribute output res
    mp_output_res = [[] for _ in range(args.num_process)]
    for i, o in enumerate(output):
        mp_output_res[i % args.num_process].append(o)

    for i, mp_output_file in enumerate(mp_output_files):
        with open(mp_output_file, 'w') as f:
            for o in mp_output_res[i]:
                f.write(json.dumps(o) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_process', type=int, required=True)

    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--original_output_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--num_test', type=int, default=None)
    args = parser.parse_args()
    main(args)
