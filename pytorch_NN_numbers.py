import torch.utils
import torch.utils.data
import file_reader
import random
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def load():

    # trainingimages = torch.Tensor(file_reader.faceReader("facedatatrain"))
    # traininglabels = torch.LongTensor(file_reader.face_labelReader("facedatatrainlabels"))
    # training_dataset = TensorDataset(trainingimages, traininglabels)

    # testimages = torch.Tensor(file_reader.faceReader("facedatatest"))
    # testlabels = torch.LongTensor(file_reader.face_labelReader("facedatatestlabels"))
    # test_dataset = TensorDataset(testimages, testlabels)

    trainingimages = torch.Tensor(file_reader.numberReader("trainingimages"))
    traininglabels = torch.LongTensor(file_reader.number_labelReader("traininglabels"))
    training_dataset = TensorDataset(trainingimages, traininglabels)

    testimages = torch.Tensor(file_reader.numberReader("testimages"))
    testlabels = torch.LongTensor(file_reader.number_labelReader("testlabels"))
    test_dataset = TensorDataset(testimages, testlabels)

    #print(trainingimages.shape[1])

    return test_dataset, training_dataset

class Digit_classifier(nn.Module): #child of model

    def __init__(self, input_size):

        super(Digit_classifier, self).__init__()
        
        #initializes the elements needed for the forward funciton

        self.hidden_1 = nn.Linear(input_size, 128)
        self.hidden_2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

        self.activation = nn.Sigmoid()
        
    def forward(self, x):

        #basically defining how the forward function would work in sequence

        x = self.hidden_1(x)
        x = self.activation(x)
        x = self.hidden_2(x)
        x = self.activation(x)
        x = self.output(x)
        
        return x


def train(model, train_loader, loss_function, optimizer, epochs):
    print("Starting training")

    for epoch in range(epochs): #passes over the images these many times
        
        print("epoch: ", epoch)

        model.train()

        for images, labels in train_loader: #passes over the images in batch size

            optimizer.zero_grad() #resets derivatives from last iteration

            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step() #subtracts old weight by derivative times the learning rate

    print("done training")

def eval(model, test_loader, test_dataset):
    model.eval()
    correct = 0

    for images, labels in test_loader: #goes through data in batch

        model_prediction = model(images) #10 different prob for each image
        #print(model_prediction)

        max, idx = torch.max(model_prediction, dim = 1) #takes max of the prob from the list of probs

        for i in range(len(idx)):
            if(labels[i] == idx[i]):
                correct += 1


    print("correct: ", correct, " out of: ", len(test_dataset), " accuracy: ", correct/len(test_dataset))

    return correct/len(test_dataset)

if __name__ == "__main__":

    training_dataset, test_dataset = load()

    train_batch_size = 20

    train_loader = DataLoader(training_dataset, batch_size=train_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=train_batch_size)

    input_size = training_dataset.tensors[0].shape[1] #length of the image converted to a list
    model = Digit_classifier(input_size) #initializes the actual model

    loss_function = nn.CrossEntropyLoss() #initializes the loss function 
    optimizer = torch.optim.Adam(model.parameters()) #the optimizer changes the weights depending on the outcome

    epochs = 15
    train(model, train_loader, loss_function, optimizer, epochs)
    
    eval(model, test_loader, test_dataset)

