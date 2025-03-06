from transformers import AutoConfig, AutoTokenizer
from model_qwen import GMQwen2ForCausalLM
from peft import get_peft_model, LoraConfig, TaskType


def preprocess(model_name_or_path: str, output_dir: str):
    """Preprocess the Llama-3.1 checkpoint into a gated-memory model checkpoint.
    Args:
        model_name_or_path (str): The Llama-3.1 checkpoint dir.
        output_dir (str): The saving dir.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
    model_config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
    model = GMQwen2ForCausalLM.from_pretrained(model_name_or_path, config=model_config, local_files_only=True, low_cpu_mem_usage=False, _fast_init=False)
    lora_config = LoraConfig(
        r=1,
        target_modules=["lm_head"],
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=0.0,
        modules_to_save=[f"layers.{i}.self_attn.mem_proj" for i in range(len(model.model.layers))]+[f"layers.{i}.self_attn.gate_proj" for i in range(len(model.model.layers))],
    )
    model = get_peft_model(model, lora_config)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    preprocess("/ceph/home/muhan01/huggingfacemodels/Qwen2.5-14B-Instruct", "models/LinGated-Memory-Qwen2.5-14B-Instruct")
