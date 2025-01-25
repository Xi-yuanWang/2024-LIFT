import csv
import os

csv_data = []
for filename in os.listdir('outputs/detail'): 
    # modify here to control included file
    if not filename.startswith('pissa_') and 'syn_qa' not in filename:
        continue
    # ====================================

    with open(os.path.join('outputs/detail', filename), 'r') as f:
        lines = f.readlines()

    start_pos = -1
    for i, line in enumerate(lines):
        if i + 30 > len(lines) and 'Fail                : All       ' in line:
            start_pos = i
            break

    data = lines[start_pos:]
    data = [d.strip().split()[-1] for d in data if 'examples' not in d and d.strip() != '']

    if len(data) == 10:
        data.insert(2, '\\')
        data.extend(['' for _ in range(8)])
        data[0] = '\\'
        data[5] = '\\'
        data[7] = '\\'
        data[8] = '\\'
        data[9] = '\\'
    data.insert(0, filename)
    csv_data.append(data)

    # for d in data:
    #     print(d)
    # print(len(data))
    # input()


# transpose 
csv_data = list(zip(*csv_data))

with open('test.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)