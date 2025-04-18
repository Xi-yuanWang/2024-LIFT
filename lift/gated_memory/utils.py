from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig
from model_qwen import GMQwen2ForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import torch

def preprocess(model_name_or_path: str, output_dir: str):
    """Preprocess the Llama-3.1 checkpoint into a gated-memory model checkpoint.
    Args:
        model_name_or_path (str): The Llama-3.1 checkpoint dir.
        output_dir (str): The saving dir.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
    model_config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
    '''
    quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
    '''
    model = GMQwen2ForCausalLM.from_pretrained(model_name_or_path, config=model_config, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=False, _fast_init=False) # quantization_config=quantization_config
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    preprocess("/ceph/home/muhan01/huggingfacemodels/Qwen2.5-7B-Instruct-YaRN", "/ceph/home/muhan01/huggingfacemodels/GM6Qwen2.5-7B-Instruct-YaRN")
