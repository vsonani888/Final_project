# import matplotlib.pyplot as plt
# from parse import extract_accuracy

# # Load parsed data
# train_sizes, pf_data, mn_data, pt_data = extract_accuracy()

# # Unpack into values and std devs
# pf_acc, pf_std = zip(*pf_data)
# mn_acc, mn_std = zip(*mn_data)
# pt_acc, pt_std = zip(*pt_data)

# # Plot
# plt.figure(figsize=(10, 6))
# plt.errorbar(train_sizes, pf_acc, yerr=pf_std, label="Perceptron Face", capsize=5, marker='o')
# plt.errorbar(train_sizes, mn_acc, yerr=mn_std, label="Manual NN Face", capsize=5, marker='s')
# plt.errorbar(train_sizes, pt_acc, yerr=pt_std, label="PyTorch NN Face", capsize=5, marker='^')

# plt.title("Learning Curve: Face Classification Accuracy")
# plt.xlabel("% of Training Data Used")
# plt.ylabel("Accuracy")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
import matplotlib.pyplot as plt
from parse import extract_times

train_sizes, pf_times, mn_times, pt_times = extract_times()

pf_mean, pf_std = zip(*pf_times)
mn_mean, mn_std = zip(*mn_times)
pt_mean, pt_std = zip(*pt_times)

plt.figure(figsize=(10, 6))
plt.errorbar(train_sizes, pf_mean, yerr=pf_std, label="Perceptron Face", capsize=5, marker='o')
plt.errorbar(train_sizes, mn_mean, yerr=mn_std, label="Manual NN Face", capsize=5, marker='s')
plt.errorbar(train_sizes, pt_mean, yerr=pt_std, label="PyTorch NN Face", capsize=5, marker='^')

plt.title("Training Time vs. Training Set Size")
plt.xlabel("% of Training Data Used")
plt.ylabel("Training Time (seconds)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
