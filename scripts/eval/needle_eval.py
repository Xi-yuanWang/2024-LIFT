import os
import openai
import tqdm
import json
import argparse

with open('api_key.txt', 'r') as f:
    api_key = f.read().strip()
client = openai.Client(api_key=api_key)

def get_gpt4_score(question: str, reference: str, pred: str):
    sys_prompt = "Given one question, there is a groundtruth and a predict_answer. Please decide whether they are the same or not in semantic. Please only output 'True' or 'False' ."

    prompt = [{"role": "system", "content": sys_prompt,},
    {
        "role": "user",
        "content": "Question: "
        + question
        + "\n"
        + "groundtruth = "
        + reference
        + "\n"
        + "predict_answer = "
        + pred,
    }]
    response = client.chat.completions.create(
        model="gpt-4-0613",
        messages=prompt,
        max_tokens=10,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    response = response.choices[0].message.content
    if not response:
        print("Error: No response from GPT-4")
    if 'True' in response or 'true' in response or 'TRUE' in response:
        return True
    else:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the input JSON/JSONL file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output JSONL file.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the output JSON file.')
    parser.add_argument('--prompt', type=str, default='What is the best thing to do in San Francisco?')
    parser.add_argument('--answer', type=str, default='The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.')
    args = parser.parse_args()
    
    file_name, file_ext = os.path.splitext(args.input)
    if file_ext == '.json':
        with open(args.input, 'r') as f:
            samples = json.load(f)
    elif file_ext == '.jsonl':
        samples = []
        with open(args.input, 'r') as f:
            for line in f:
                samples.append(json.loads(line))
    
    # overwrite the output file
    if args.overwrite and os.path.exists(args.output):
        os.remove(args.output)

    # resume the output file
    gpt4_scores = []
    num_resumed = 0
    if os.path.exists(args.output):
        num_resumed = len(open(args.output, 'r').readlines())
    
    for sample in tqdm.tqdm(samples[num_resumed:], initial=num_resumed, total=len(samples), desc='Calculating GPT-4 score'):
        sample['gpt4_score'] = get_gpt4_score(args.prompt, args.answer, sample['pred'])
        with open(args.output, 'a') as f:
            f.write(json.dumps(sample) + '\n')
