import argparse
import json
import os
from openai import OpenAI
from typing import Dict
from datetime import datetime


def parse_metadata(metadata_kwargs) -> Dict:
    if metadata_kwargs is None:
        return {}
    if len(metadata_kwargs) % 2 == 1:
        raise ValueError("Fail to match metadata_kwargs in pairs.")
    return {key: value for key, value in zip(metadata_kwargs[::2], metadata_kwargs[1::2])}


def create_batch_file(client: OpenAI, input_path: str, metadata: Dict):
    temp_file = '_temp_' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.jsonl'
    if os.path.exists(temp_file):
        raise FileExistsError("Evaluation program tries to write the file `_temp.jsonl` as a temporary file but finds it exists.")
    with open(input_path, 'r') as f:
        data = [json.loads(l) for l in f]
    with open(temp_file, 'w') as f:
        PROMPT_TEMPLATE = "Question: {question}\ngroundtruth = {reference}\npredict_answer = {pred}"
        for i, d in enumerate(data):
            if 'sample' in d:
                d = d['sample']
            for j, qa_pair in enumerate(d['qa_pairs']):
                messages = [
                    {'role': 'system', 'content': "Given one question, there is a groundtruth and a predict_answer. Please decide whether they are the same or not in semantic. Please only output 'True' or 'False' ."},
                    {'role': 'user', 'content': PROMPT_TEMPLATE.format(question=qa_pair['Q'], reference=qa_pair['A'], pred=qa_pair['pred'])}
                ]
                entry = {
                    'custom_id': f'LooGLE-{i}-{j}',
                    'method': 'POST',
                    'url': '/v1/chat/completions',
                    'body': {
                        'model': 'gpt-4-0613',
                        'messages': messages,
                        'max_tokens': 10
                    }
                }
                f.write(json.dumps(entry) + '\n')
    print("Write to temp file. Submitting...")
    
    batch_input_file = client.files.create(
        file=open(temp_file, 'rb'),
        purpose='batch'
    )

    batch_input_file_id = batch_input_file.id
    batch_info = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint='/v1/chat/completions',
        completion_window='24h',
        metadata=metadata
    )

    print("Finish creating and submitting a Batch object.")
    print("File object:")
    print(batch_input_file)
    print("Batch object:")
    print(batch_info)
    print(f"Please remember the ID attr. Batch ID = {batch_info.id}")
    os.remove(temp_file)


def retrieve_batch(client: OpenAI, batch_id, file_id, loogle_file, result_file):
    os.makedirs(os.path.split(result_file)[0], exist_ok=True)
    if batch_id is not None:
        info = client.batches.retrieve(batch_id)
        print("Retrieved info:")
        print(info)
        if info.status == 'completed':
            print(f"Batch completed. Retrieving results from `{info.output_file_id}`...")
            f_result = client.files.content(info.output_file_id)
            print("Processing...")
            data = f_result.read().decode()
            data = [d for d in data.split('\n') if len(d) > 0]
            data = [json.loads(l) for l in data]
            data = {d['custom_id']: d for d in data}
            with open(loogle_file, 'r') as f:
                raw_data = [json.loads(l) for l in f]
            for i, d in enumerate(raw_data):
                if 'sample' in d:
                    for j, q in enumerate(d['sample']['qa_pairs']):
                        response = data[f'LooGLE-{i}-{j}']['response']['body']['choices'][0]['message']['content']
                        q['score'] = 'true' in response.lower()
                    d['meta_data']['score'] = sum([q['score'] for q in d['sample']['qa_pairs']]) / len(d['sample']['qa_pairs'])
                else:
                    for j, q in enumerate(d['qa_pairs']):
                        response = data[f'LooGLE-{i}-{j}']['response']['body']['choices'][0]['message']['content']
                        q['score'] = 'true' in response.lower()
            with open(result_file, 'w') as f:
                for d in raw_data:
                    f.write(json.dumps(d) + '\n')
            print(f"Write to {result_file}. Deleting OpenAI input and output files...")
            client.files.delete(info.input_file_id)
            client.files.delete(info.output_file_id)
            print("Done.")
        elif info.error_file_id is not None:
            f_error = client.files.content(info.error_file_id)
            print(f_error.read().decode())
    elif file_id is not None:
        info = client.files.retrieve(file_id)
        print("Retrieved info:")
        print(info)
        f_result = client.files.content(file_id)
        data = f_result.read().decode()
        with open(result_file, 'w') as f:
            f.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', required=True, choices=['submit', 'list', 'retrieve'])
    parser.add_argument('--api_key', default=os.environ["GPT4_API_KEY"], help="GPT4 API KEY. Defaults to the envar `GPT4_API_KEY`.")
    parser.add_argument('--loogle_file', help="LooGLE output file.")
    parser.add_argument('--batch_id', help="The batch ID of a previous request.")
    parser.add_argument('--file_id', help="The output file ID.")
    parser.add_argument('--result_file', help="The result GPT_SCORE file.")
    parser.add_argument('--metadata_kwargs', nargs='*', help="The following are metadata kwargs.")
    args = parser.parse_args()
    metadata = parse_metadata(args.metadata_kwargs)

    client = OpenAI(api_key=args.api_key)

    if args.action == 'submit':
        if args.loogle_file is None:
            raise ValueError("Please provide `--loogle_file`!")
        create_batch_file(client, args.loogle_file, metadata)

    elif args.action == 'list':
        file_list = list(client.files.list())
        if len(file_list) == 0:
            print("Empty!")
        else:
            for i, entry in enumerate(file_list):
                print(f"[{i}]")
                print(entry)
    
    else:
        if (args.batch_id is None and args.file_id is None) or args.result_file is None or args.loogle_file is None:
            raise ValueError("Please provide `--batch_id` or `--file_id`, `--loogle_file`, and `--result_file`!")
        if args.batch_id is not None and args.file_id is not None:
            raise ValueError("Only one of `--batch_id` and `--file_id` should be specified.")
        retrieve_batch(client, args.batch_id, args.file_id, args.loogle_file, args.result_file)


if __name__ == '__main__':
    main()
