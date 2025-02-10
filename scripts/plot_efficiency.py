from lift.model import load_tokenizer
from transformers import AutoTokenizer
import json
import numpy as np
import matplotlib.pyplot as plt

def generate_line(data, is_baseline=False):
    line = []
    for sample in data:
        line.append((sample['length'], (sample['time'] / 20 if is_baseline else sample['time']) / 1000))
    return line

def fit_line(data, poly=1, is_baseline=False):
    xs = [sample['length'] for sample in data]
    ys = [(sample['time'] / 20 if is_baseline else sample['time']) / 1000 for sample in data]
    
    coefficients = np.polyfit(xs, ys, poly)
    poly = np.poly1d(coefficients)

    x_fit = np.concat([np.linspace(0, 1000000, 100), np.array([90100])], axis=0)
    y_fit = poly(x_fit)
    x_fit, crossx = x_fit[:-1], x_fit[-1]
    y_fit, crossy = y_fit[:-1], y_fit[-1]
    return x_fit, y_fit, crossx, crossy

tokenizer = load_tokenizer('models/tokenizer')
baseline_path = 'outputs/Efficiency-Baseline.jsonl'
pissa_path = 'outputs/Efficiency-PiSSA-rICL-C1M3-ICL.jsonl'
gate_path = 'outputs/Efficiency-Gate-rICL-C3M5-ICL.jsonl'

baseline_data = [json.loads(l) for l in open(baseline_path, 'r')]
pissa_data = [json.loads(l) for l in open(pissa_path, 'r')]
gate_data = [json.loads(l) for l in open(gate_path, 'r')]

baseline_line = generate_line(baseline_data, is_baseline=True)
pissa_line = generate_line(pissa_data)
gate_line = generate_line(gate_data)

# 提取 x 和 y 数据
baseline_x, baseline_y = [b[0] for b in baseline_line], [b[1] for b in baseline_line]
pissa_x, pissa_y = [p[0] for p in pissa_line], [p[1] for p in pissa_line]
gate_x, gate_y = [g[0] for g in gate_line], [g[1] for g in gate_line]
baseline_xf, baseline_yf, crossx, crossy = fit_line(baseline_data, poly=2, is_baseline=True)
pissa_xf, pissa_yf, _, _ = fit_line(pissa_data, poly=1)
gate_xf, gate_yf, _, _ = fit_line(gate_data, poly=1)

# 创建图形
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(10, 6))

# 绘制三条曲线
plt.plot(baseline_x, baseline_y, label="Baseline", color='#4485C7', marker='o', linewidth=4)
plt.plot(pissa_x, pissa_y, label="PiSSA", color='#DBB428', marker='o', linewidth=4)
plt.plot(gate_x, gate_y, label="Gated Memory", color='#D4562E', marker='o', linewidth=4)
plt.plot(baseline_xf, baseline_yf, label="Fitted Baseline", color='#79B0E9', linestyle='--', linewidth=2)
plt.plot(pissa_xf, pissa_yf, label="Fitted PiSSA", color='#F1D77D', linestyle='--', linewidth=2)
plt.plot(gate_xf, gate_yf, label="Fitted Gated Memory", color='#EA8D6E', linestyle='--', linewidth=2)
plt.scatter([crossx], [crossy], color='red', marker='x', s=100)


# 添加标题和标签
# plt.title('Efficiency Comparison')
plt.xlabel('length / token')
plt.ylabel('average time per token / sec')

# 添加图例
plt.legend()

# 显示图形
plt.savefig('efficiency.pdf')
plt.savefig('efficiency.png')