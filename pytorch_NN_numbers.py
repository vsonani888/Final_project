import file_reader
import random
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

if __name__ == "__main__":
    print("main in manual neural network")

    #reading all files

    trainingimages = file_reader.numberReader("trainingimages")
    traininglabels = file_reader.number_labelReader("traininglabels")

    validationimages = file_reader.numberReader("validationimages")
    validationlabels = file_reader.number_labelReader("validationlabels")

    testimages = file_reader.numberReader("testimages")
    testlabels = file_reader.number_labelReader("testlabels")

