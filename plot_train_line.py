
import matplotlib.pyplot as plt
import ast

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'

with open('./ckpt/train_log_val.txt', 'r') as file:
    lines = file.readlines()

datasets = [
    "Brain", "Breast", "CTC", "Colorectal", "Head_Neck", 
    "Liver", "Lung", "PBMCs", "Pelvic_cavity", "cell_line"
]

val_acc = []
for line in lines:
    if "val acc:" in line:
        acc_values = [float(val.split(': ')[1]) for val in line.split('\t')]
        val_acc.append(acc_values)

val_acc = list(zip(*val_acc))

plt.figure(figsize=(8, 6))
for i, acc_list in enumerate(val_acc):
    accuracies = (0,) + acc_list  
    plt.plot(range(0, len(accuracies)), accuracies, label=datasets[i])

plt.title('Validation Accuracy Across Tissues in Training', fontsize=16)
plt.xlim(0,55)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Validation Accuracy', fontsize=16)
plt.legend(loc='lower right')
plt.grid(True)
ax = plt.gca()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(False)

plt.savefig("./results/val_acc.pdf")
plt.show()

file_path = './ckpt/train_log_loss.txt'
data_list = []

with open(file_path, 'r') as file:
    for line in file:
        data_dict = ast.literal_eval(line.strip())
        data_list.append(data_dict)

loss_values = [entry['loss'] for entry in data_list]
cancer_mean_values = [entry['cancer_mean'] for entry in data_list]
tissue_mean_values = [entry['tissue_mean'] for entry in data_list]
ad_mean_values = [entry['ad_mean'] for entry in data_list]
reconstruct_mean_values = [entry['reconstruct_mean'] for entry in data_list]

plt.figure(figsize=(8, 6))
plt.plot(loss_values, label='Total Loss')
plt.plot(cancer_mean_values, label='Identification Loss')
plt.plot(tissue_mean_values, label='Routing Loss')
plt.plot(ad_mean_values, label='Domain Adversarial Loss')
plt.plot(reconstruct_mean_values, label='Reconstruction Loss')

plt.xlabel('Step', fontsize=16)
plt.ylabel('Value', fontsize=16)
plt.title('Loss Changed in Training', fontsize=16)
plt.legend()
plt.grid(True)
ax = plt.gca()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(False)

plt.savefig("./results/training_loss.pdf")
plt.show()