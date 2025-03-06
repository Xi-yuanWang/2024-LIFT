"""
Test LIFT on LooGLE (both ShortQA and LongQA, default).
No random ICL.
Allow:
- Instruct or Base, controlled by --use_chat_template.
- Random segment or not, controlled by --use_random_segment.
"""
from transformers import (
    TrainingArguments,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    BitsAndBytesConfig,
)
from lift.args import (
    ModelArguments,
    DataTrainingArguments,
    CustomTrainingArguments,
    parse_args
)
from lift.context_dataset import ContextDataset, RandomContextDataset
from lift.model import load_tokenizer, load_model
from lift.train import train
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Literal
from numpy.random import randint
from nltk import sent_tokenize
import logging
import json
import os
import torch
import tqdm


LOOGLE_CHAT = "The article {title}: \n{input}\nPlease answer the question based on {title}.\nQuestion: {question}\nAnswer: "
LOOGLE_NON_CHAT = {
    'pre-inst': "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\nYou are given an article {title}. Please answer the question based on {title}. Question: {question}.\n\n### Input:\n{input}\n\n### Response:\n",
    'post-inst': "Below is an input that provides part of the article {title}, paired with an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n### Input:\n{input}\n### End of Input\n\n### Instruction:\nPlease answer the question based on {title}. Question: {question}.\n### End of Instruction\n\n### Response:\n"
}


@dataclass
class TestArguments:
    input_file: str = field(
        default=None,
        metadata={"help": "The input file for the test."}
    )
    output_file: str = field(
        default=None,
        metadata={"help": "The output file for the test."}
    )
    overwrite: bool = field(
        default=False,
        metadata={"help": "Overwrite the output file."}
    )
    num_syn_qa: int = field(
        default=0,
        metadata={"help": "The number of auxiliary tasks (synthetic QAs)."}
    )
    generator_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The generator model name or path. Required if num_syn_qa > 0."}
    )
    num_test: Optional[int] = field(
        default=None,
        metadata={'help': "Test only the first several articles."}
    )
    use_chat_template: bool = field(
        default=False,
        metadata={'help': "Apply chat template to the auxiliary tasks and LooGLE test tasks."}
    )
    post_instruction: bool = field(
        default=False,
        metadata={'help': "Use post-instruction. By default, we use pre-instruction."}
    )

    def __post_init__(self):
        if self.post_instruction and self.use_chat_template:
            self.post_instruction = False
            logging.warning("Post-instruction is only available in non-chat setting. Setting post_instruction to False.")
        if not self.post_instruction:
            raise ValueError("Pre-instruction is deprecated. Please set --post_instruction to True.")


def load_dataset(
    context: str = None,
    title: str = None,
    tokenizer: PreTrainedTokenizer = None,
    model_max_length: int = 7800,
    block_size: int = 256,
    len_segment: int = 8,
    len_offset: int = 3,
    num_syn_qa: int = 0,
    generator_name_or_path: Optional[str] = None,
    use_chat_template: bool = False,
    post_instruction: bool = True,
    use_random_segment: bool = False,
):
    base_class = RandomContextDataset if use_random_segment else ContextDataset

    class LooGLEDataset(base_class):
        def __init__(
            self,
            context: str,
            title: str,
            tokenizer: PreTrainedTokenizer,
            model_max_length: int = 7800,
            block_size: int = 256,
            len_segment: int = 8,
            len_offset: int = 3,
            num_syn_qa: int = 0,
            generator_name_or_path: Optional[str] = None,
            use_chat_template: bool = False,
            post_instruction: bool = False,
        ):
            context = title + '\n' + context
            super().__init__(
                context=context,
                tokenizer=tokenizer,
                model_max_length=model_max_length,
                block_size=block_size,
                len_segment=len_segment,
                len_offset=len_offset,
            )
            # Generate QA pairs
            self.qa_data = []
            self.num_syn_qa = num_syn_qa
            if num_syn_qa > 0:
                context_sent = sent_tokenize(context)
                assert len(context_sent) >= 16, "The length of the context should be at least 25 sentences."
                generator = AutoModelForCausalLM.from_pretrained(
                    generator_name_or_path,
                    device_map='auto',
                    torch_dtype=torch.bfloat16,
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type='nf4',
                    ),
                )
                gen_tokenizer = load_tokenizer(generator_name_or_path)
                generator.eval()
                while len(self.qa_data) < num_syn_qa:
                    result = self.generate_task(
                        generator=generator,
                        gen_tokenizer=gen_tokenizer,
                        full_context=context,
                        context_sent=context_sent,
                        title=title,
                        model_max_length=model_max_length,
                        use_chat_template=use_chat_template,
                        post_instruction=post_instruction,
                    )
                    if result is not None:
                        self.qa_data.append(result)
            self.enable_qa_tag = False
        
        @torch.no_grad()
        def generate_task(
            self,
            generator: PreTrainedModel,
            gen_tokenizer: PreTrainedTokenizer,
            full_context: str,
            context_sent: List[str],
            title: str,
            model_max_length: int = 7800,
            use_chat_template: bool = False,
            post_instruction: bool = False,
        ):
            st_pos = randint(0, len(context_sent) - 16)
            context = ' '.join(context_sent[st_pos:st_pos+16])
            messages = [
                {
                    'role': "system",
                    'content': "You are a helpful assistant."
                },
                {
                    'role': "user", 
                    'content': f"You are given a piece of text as the context. You should generate ONLY one question and the corresponding answer according to the context. You should also select one or more sentences directly from the original context as the evidence. The evidences must be verbatim sentences from the context. Please answer in the following format: \nQuestion: [question] \nAnswer: [answer] \nEvidence: [evidence]\nPlease DON'T output quotes when outputting evidences. The following is the piece of text: {context}"
                }
            ]
            input_ids = gen_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(generator.device)
            mask_attention = torch.ones_like(input_ids)
            for _ in range(5):
                outputs = generator.generate(
                    input_ids=input_ids,
                    attention_mask=mask_attention.to(generator.device),
                    max_new_tokens=1024,
                    pad_token_id=gen_tokenizer.eos_token_id,
                    eos_token_id=gen_tokenizer.eos_token_id,
                    do_sample=True,
                )
                response = gen_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
                question_position = response.find("Question:")
                answer_position = response.find("Answer:")
                evidence_position = response.find("Evidence:")
                if question_position == -1 or answer_position == -1 or evidence_position == -1:
                    continue
                question = response[question_position + 9:answer_position].strip()
                answer = response[answer_position + 7:evidence_position].strip()
                evidence = response[evidence_position + 9:].strip()
                if evidence not in context:
                    continue
                break
            else:
                logging.warning("Fail to generate a QA pair, skip.")
                return None
            
            if use_chat_template:
                messages = [
                    {'role': 'system', 'content': "You are a helpful assistant."},
                    {'role': 'user', 'content': LOOGLE_CHAT.format(title=title, input=full_context, question=question)},
                    {'role': 'assistant', 'content': answer},
                ]
                input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False)
                input_length = len(self.tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True))
            else:
                template = LOOGLE_NON_CHAT['post-inst' if post_instruction else 'pre-inst']
                instruction_text = template.format(title=title, input=full_context, question=question)
                input_text = instruction_text + answer + "\n### End of Response"
                input_ids = [self.tokenizer.bos_token_id] + self.tokenizer(input_text, add_special_tokens=False)['input_ids'] + [self.tokenizer.eos_token_id]
                input_length = len(self.tokenizer(instruction_text, add_special_tokens=False)['input_ids']) + 1

            mixin = self.tokenizer("...", add_special_tokens=False)['input_ids']
            output_length = len(input_ids) - input_length
            if len(input_ids) > model_max_length:
                input_ids = input_ids[:model_max_length//2 - len(mixin)] + mixin + input_ids[-model_max_length//2:]
                input_length = len(input_ids) - output_length
            return (input_ids, input_length)
        
        def enable_qa(self):
            self.enable_qa_tag = True
            
        def disable_qa(self):
            self.enable_qa_tag = False
        
        def __getitem__(self, index):
            return super().__getitem__(index) if index < self.num_segments else self.preprocessing(self.qa_data[index - self.num_segments])
        
        def __len__(self):
            return self.num_segments + self.num_syn_qa if self.enable_qa_tag else self.num_segments
    
    return LooGLEDataset(
        context=context,
        title=title,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        block_size=block_size,
        len_segment=len_segment,
        len_offset=len_offset,
        num_syn_qa=num_syn_qa,
        generator_name_or_path=generator_name_or_path,
        use_chat_template=use_chat_template,
        post_instruction=post_instruction,
    )

    
def LooGLEtrain(
    context: str = None,
    title: str = None,
    tokenizer: PreTrainedTokenizer = None,
    num_syn_qa: int = 0,
    generator_name_or_path: Optional[str] = None,
    use_chat_template: bool = False,
    post_instruction: bool = False,
    training_args: TrainingArguments = None,
    model_name_or_path: str = None,
    model_max_length: int = 7800,
    block_size: int = 256,
    len_segment: int = 8,
    len_offset: int = 3,
    use_random_segment: bool = False,
    use_lora: bool = False,
    lora_rank: Optional[int] = None,
    lora_target_modules: Union[Literal['all-linear'], List[str]] = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    use_pissa: bool = False,
    load_in_4bit: bool = False,
    involve_qa_epochs: int = 0,
    gather_batches: bool = True,
    use_gated_memory: bool = False,
    use_prefix_tuning: bool = False,
    num_virtual_tokens: Optional[int]=None,
    **kwargs
):
    model = load_model(
        model_name_or_path=model_name_or_path,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_target_modules=lora_target_modules,
        use_pissa=use_pissa,
        load_in_4bit=load_in_4bit,
        vocab_size=len(tokenizer),
        use_gated_memory=use_gated_memory,
        use_prefix_tuning=use_prefix_tuning,
        num_virtual_tokens=num_virtual_tokens
    )
    dataset = load_dataset(
        context=context,
        title=title,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        block_size=block_size,
        len_segment=len_segment,
        len_offset=len_offset,
        num_syn_qa=num_syn_qa,
        generator_name_or_path=generator_name_or_path,
        use_chat_template=use_chat_template,
        post_instruction=post_instruction,
        use_random_segment=use_random_segment,
    )
    if use_lora or use_gated_memory or use_prefix_tuning:
        model = train(model, dataset, tokenizer, training_args, involve_qa_epochs, gather_batches)[0]
    return model


def prediction(
    data: List[Dict],
    training_args: TrainingArguments,
    lift_args: Dict,
    output_file: str,
    num_resumed: int = 0,
    num_syn_qa: int = 0,
    generator_name_or_path: Optional[str] = None,
    use_chat_template: bool = False,
    post_instruction: bool = False,
):
    tokenizer = load_tokenizer(lift_args['tokenizer_name_or_path'])
    mixin = tokenizer("...", add_special_tokens=False)['input_ids']
    model_max_length = lift_args['model_max_length']
    
    for i, sample in enumerate(tqdm.tqdm(data, desc="Sample")):
        if i < num_resumed:
            continue
        context = sample['input']
        title = sample['title']
        qa_pairs = eval(sample['qa_pairs'])
        model = LooGLEtrain(
            context=context,
            title=title,
            tokenizer=tokenizer,
            num_syn_qa=num_syn_qa,
            generator_name_or_path=generator_name_or_path,
            use_chat_template=use_chat_template,
            post_instruction=post_instruction,
            training_args=training_args,
            **lift_args,
        )
        model.eval()
        for qa_pair in tqdm.tqdm(qa_pairs, desc="QA Pair"):
            if use_chat_template:
                messages = [
                    {'role': 'system', 'content': "You are a helpful assistant."},
                    {'role': 'user', 'content': LOOGLE_CHAT.format(title=title, input=context, question=qa_pair['Q'])},
                ]
                input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            else:
                template = LOOGLE_NON_CHAT['post-inst' if post_instruction else 'pre-inst']
                input_text = template.format(title=title, input=context, question=qa_pair['Q'])
                input_ids = [tokenizer.bos_token_id] + tokenizer(input_text, add_special_tokens=False)['input_ids']

            if len(input_ids) > model_max_length:
                input_ids = input_ids[:model_max_length//2 - len(mixin)] + mixin + input_ids[-model_max_length//2:]
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=200,
                use_cache=True,
                do_sample=False,
            )
            response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
            if "### End of Response" in response:
                response = response[:response.find("### End of Response")]
            qa_pair['pred'] = response
        output_case = {
            'title': title,
            'input': context,
            'qa_pairs': qa_pairs,
        }
        with open(output_file, 'a') as f:
            f.write(json.dumps(output_case) + '\n')


def main():
    training_args, test_args, lift_args = parse_args(
        (TrainingArguments, TestArguments, (ModelArguments, CustomTrainingArguments, DataTrainingArguments)),
        no_dict=(TrainingArguments,)
    )
    input_file = test_args.pop('input_file')
    output_file = test_args.pop('output_file')
    overwrite = test_args.pop('overwrite')
    num_test = test_args.pop('num_test')

    num_resumed = 0
    if os.path.exists(output_file):
        if overwrite:
            os.remove(output_file)
        else:
            with open(output_file, 'r') as f:
                num_resumed = len(f.readlines())
    with open(input_file, 'r') as f:
        input_data = [json.loads(line) for line in f]
    if num_test is not None:
        input_data = input_data[:num_test]
    prediction(input_data, training_args, lift_args, output_file, num_resumed=num_resumed, **test_args)


if __name__ == '__main__':
    main()
