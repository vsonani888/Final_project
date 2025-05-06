import re
def extract_times(filename="message.txt"):
    with open(filename, "r") as f:
        lines = f.readlines()

    # Skip header
    lines = lines[1:]

    # Only keep training time rows (even lines after header)
    time_lines = [line.strip() for i, line in enumerate(lines) if i % 2 == 0]

    train_sizes = []
    perceptron_face_times = []
    perceptron_num_times = []
    manual_face_times = []
    manual_num_times = []
    pytorch_face_times = []
    pytorch_num_times = []

    for i, line in enumerate(time_lines):
        parts = re.findall(r"([\d.]+) ± ([\d.]+)", line)
        if len(parts) == 6:
            train_sizes.append((i + 1) * 10)

            perceptron_face_times.append((float(parts[0][0]), float(parts[0][1])))
            perceptron_num_times.append((float(parts[1][0]), float(parts[1][1])))
            manual_face_times.append((float(parts[2][0]), float(parts[2][1])))
            manual_num_times.append((float(parts[3][0]), float(parts[3][1])))
            pytorch_face_times.append((float(parts[4][0]), float(parts[4][1])))
            pytorch_num_times.append((float(parts[5][0]), float(parts[5][1])))

    return train_sizes, perceptron_face_times, manual_face_times, pytorch_face_times

# def extract_accuracy(filename="message.txt"):
#     with open(filename, "r") as f:
#         lines = f.readlines()

#     # Skip header
#     lines = lines[1:]

#     # Only keep accuracy lines (odd rows after header)
#     accuracy_lines = [line.strip() for i, line in enumerate(lines) if i % 2 == 1]

#     train_sizes = []
#     perceptron_face_acc = []
#     perceptron_num_acc = []
#     manual_face_acc = []
#     manual_num_acc = []
#     pytorch_face_acc = []
#     pytorch_num_acc = []

#     for i, line in enumerate(accuracy_lines):
#         parts = re.findall(r"([\d.]+) ± ([\d.]+)", line)
#         if len(parts) == 6:
#             train_sizes.append((i + 1) * 10)

#             perceptron_face_acc.append((float(parts[0][0]), float(parts[0][1])))
#             perceptron_num_acc.append((float(parts[1][0]), float(parts[1][1])))
#             manual_face_acc.append((float(parts[2][0]), float(parts[2][1])))
#             manual_num_acc.append((float(parts[3][0]), float(parts[3][1])))
#             pytorch_face_acc.append((float(parts[4][0]), float(parts[4][1])))
#             pytorch_num_acc.append((float(parts[5][0]), float(parts[5][1])))

#     return train_sizes, perceptron_face_acc, manual_face_acc, pytorch_face_acc
