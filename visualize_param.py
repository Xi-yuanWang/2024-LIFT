from safetensors import safe_open

tensors = {}
with safe_open("models/qwentrain.ss3.tmp/adapter_model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
        print(k, tensors[k].shape, tensors[k].mean().item(), tensors[k].std().item(), tensors[k].abs().mean().item(), tensors[k].max().item(), tensors[k].min().item())
