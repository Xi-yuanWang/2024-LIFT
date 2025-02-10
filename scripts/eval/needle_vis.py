"""
This code is adapted from https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/main/viz/CreateVizFromLLMTesting.ipynb.
"""
import os
import json
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from argparse import ArgumentParser


def load_pred_file(pred_file: str):
    with open(pred_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    if 'length' not in data[0]:  # the first entry may be the config
        return data, 1
    else:
        return data, 0


def toggle_test_cases(data: list[dict], st_point: int):
    table = {}
    for sample in data[st_point:]:
        item = (sample['length'], sample['depth'])
        if item not in table:
            table[item] = []
        table[item].append(sample['gpt4_score'])
    for key in table:
        table[key] = sum(table[key]) / len(table[key])
    return [{'depth': d, 'length': l, 'gpt4_score': v} for (l, d), v in table.items()]


def plot_data(eval_data: list[dict], output_path: str):
    table = [{'Document Depth': d['depth'], 'Context Length': d['length'], 'Score': d['gpt4_score']} for d in eval_data]
    df = pd.DataFrame(table)
    print(df.head())
    print (f"You have {len(df)} rows")
    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    print(pivot_table.iloc[:5, :5])
    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        vmin=0.0,
        vmax=1.0
    )

    # More aesthetics
    # plt.title('Pressure Testing GPT-4 128K Context\nFact Retrieval Across Context Lengths ("Needle In A HayStack")')  # Adds a title
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    # Show the plot
    plt.savefig(output_path)
    # plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument('-I', '--input', help="The path to the prediction file.")
    parser.add_argument('-O', '--output', help="The path to the plotted figure.")
    args = parser.parse_args()
    data, st_point = load_pred_file(args.input)
    eval_data = toggle_test_cases(data, st_point)
    plot_data(eval_data, args.output)


if __name__ == '__main__':
    main()
