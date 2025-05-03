import perceptron_faces
import perceptron_numbers

import manual_NN_faces
import manual_NN_numbers

import pytorch_NN_numbers

import file_reader

import time
from tabulate import tabulate
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def face_perceptron_runner(new_face_trainingimages, new_face_traininglabels, face_testimages, face_testlabels):

    num_classes=2 #tells if face or not, highest prob is the output.
    learning_rate = 0.1
    input_size = len(new_face_trainingimages[0])
    #print(input_size)

    weights = []

    for each_class in range(num_classes):
        class_weights = []

        for i in range(input_size):
            weight = random.uniform(-0.01, 0.01)
            class_weights.append(weight)
        
        weights.append(class_weights)

    biases = []

    for i in range(num_classes):
        biases.append(0.0)

    epochs = 5

    new_weights, new_bias = perceptron_faces.perceptron(new_face_trainingimages, new_face_traininglabels, weights, biases, learning_rate, epochs)

    accuracy = perceptron_faces.evaluate(face_testimages, face_testlabels, new_weights, new_bias)

    return accuracy


def num_perceptron_runner(new_num_trainingimages, new_num_traininglabels, num_testimages, num_testlabels):

    num_classes=10 #for each number 0-9, each tells if the digit is in its class or not, highest prob is the output.
    learning_rate = 0.1
    input_size = len(new_num_trainingimages[0])
    #print(input_size)

    weights = []

    for each_class in range(num_classes):
        class_weights = []

        for i in range(input_size):
            weight = random.uniform(-0.01, 0.01)
            class_weights.append(weight)
        
        weights.append(class_weights)

    biases = []

    for i in range(num_classes):
        biases.append(0.0)

    epochs = 5

    new_weights, new_bias = perceptron_numbers.perceptron(new_num_trainingimages, new_num_traininglabels, weights, biases, learning_rate, epochs)

    accuracy = perceptron_numbers.evaluate(num_testimages, num_testlabels, new_weights, new_bias)

    return accuracy

def face_manual_NN_runner(new_face_trainingimages, new_face_traininglabels, face_testimages, face_testlabels):
    input_size = len(new_face_trainingimages[0])

    output_size = 10
    learning_rate = 0.1

    hidden_1_size = 128
    hidden_2_size = 64

    weights_1 = [] #for hidden 1, size 128 * 560
    weights_2 = [] #for hidden 2, size 64 * 128
    weights_3 = [] #for hidden 3, size 10 * 64

    biases_1 = [] #for hidden 1, size 128
    biases_2 = [] #for hidden 2, size 64
    biases_3 = [] #for hidden 3, size 10

    epochs = 5

    weights_1, weights_2, weights_3, biases_1, biases_2, biases_3 = manual_NN_faces.fill(input_size, output_size, hidden_1_size, hidden_2_size, weights_1, weights_2, weights_3, biases_1, biases_2, biases_3)
    
    weights_1, weights_2, weights_3, biases_1, biases_2, biases_3 = manual_NN_faces.training(new_face_trainingimages, new_face_traininglabels, epochs, learning_rate, output_size, hidden_1_size, hidden_2_size, weights_1, weights_2, weights_3, biases_1, biases_2, biases_3)
        
    #evaluating results

    accuracy = manual_NN_faces.evalulate(face_testimages, face_testlabels, weights_1, weights_2, weights_3, biases_1, biases_2, biases_3, hidden_1_size, hidden_2_size, output_size)
    
    return accuracy


def num_manual_NN_runner(new_num_trainingimages, new_num_traininglabels, num_testimages, num_testlabels):
    input_size = len(new_num_trainingimages[0])

    output_size = 10
    learning_rate = 0.1

    hidden_1_size = 128
    hidden_2_size = 64

    weights_1 = [] #for hidden 1, size 128 * 560
    weights_2 = [] #for hidden 2, size 64 * 128
    weights_3 = [] #for hidden 3, size 10 * 64

    biases_1 = [] #for hidden 1, size 128
    biases_2 = [] #for hidden 2, size 64
    biases_3 = [] #for hidden 3, size 10

    epochs = 5

    weights_1, weights_2, weights_3, biases_1, biases_2, biases_3 = manual_NN_numbers.fill(input_size, output_size, hidden_1_size, hidden_2_size, weights_1, weights_2, weights_3, biases_1, biases_2, biases_3)
    
    weights_1, weights_2, weights_3, biases_1, biases_2, biases_3 = manual_NN_numbers.training(new_num_trainingimages, new_num_traininglabels, epochs, learning_rate, output_size, hidden_1_size, hidden_2_size, weights_1, weights_2, weights_3, biases_1, biases_2, biases_3)
        
    #evaluating results

    accuracy = manual_NN_numbers.evalulate(num_testimages, num_testlabels, weights_1, weights_2, weights_3, biases_1, biases_2, biases_3, hidden_1_size, hidden_2_size, output_size)

    return accuracy


def face_pytorch_NN_runner(new_face_trainingimages, new_face_traininglabels, face_testimages, face_testlabels):
    trainingimages = torch.Tensor(new_face_trainingimages)
    traininglabels = torch.LongTensor(new_face_traininglabels)
    training_dataset = TensorDataset(trainingimages, traininglabels)

    testimages = torch.Tensor(face_testimages)
    testlabels = torch.LongTensor(face_testlabels)
    test_dataset = TensorDataset(testimages, testlabels)

    train_batch_size = 20

    train_loader = DataLoader(training_dataset, batch_size=train_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=train_batch_size)

    input_size = training_dataset.tensors[0].shape[1] #length of the image converted to a list
    model = pytorch_NN_numbers.Digit_classifier(input_size) #initializes the actual model

    loss_function = nn.CrossEntropyLoss() #initializes the loss function 
    optimizer = torch.optim.Adam(model.parameters()) #the optimizer changes the weights depending on the outcome

    epochs = 5
    pytorch_NN_numbers.train(model, train_loader, loss_function, optimizer, epochs)

    accuracy = pytorch_NN_numbers.eval(model, test_loader, test_dataset)
    
    return accuracy

def num_pytorch_NN_runner(new_num_trainingimages, new_num_traininglabels, num_testimages, num_testlabels):
    trainingimages = torch.Tensor(new_num_trainingimages)
    traininglabels = torch.LongTensor(new_num_traininglabels)
    training_dataset = TensorDataset(trainingimages, traininglabels)

    testimages = torch.Tensor(num_testimages)
    testlabels = torch.LongTensor(num_testlabels)
    test_dataset = TensorDataset(testimages, testlabels)

    train_batch_size = 20

    train_loader = DataLoader(training_dataset, batch_size=train_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=train_batch_size)

    input_size = training_dataset.tensors[0].shape[1] #length of the image converted to a list
    model = pytorch_NN_numbers.Digit_classifier(input_size) #initializes the actual model

    loss_function = nn.CrossEntropyLoss() #initializes the loss function 
    optimizer = torch.optim.Adam(model.parameters()) #the optimizer changes the weights depending on the outcome

    epochs = 5
    pytorch_NN_numbers.train(model, train_loader, loss_function, optimizer, epochs)
    
    accuracy = pytorch_NN_numbers.eval(model, test_loader, test_dataset)

    return accuracy

def uncertainty(numbers):
    
    n = len(numbers)

    total = 0

    for num in numbers:
        total += num
    
    mean = total/n

    var = 0

    for num in numbers:
        difference = num - mean
        var += difference * difference

    variance = var / (n - 1)

    SD = variance ** 0.5
    uncertainty = SD / (n ** 0.5)

    output = f"{round(mean, 2)} ± {round(uncertainty, 2)}"
    return output

if __name__ == "__main__":
    print("main in performance")

    data = [
        [
        "% Train Data",
        "Perceptron Num Acc",
        "Manual NN Num Acc",
        "PyTorch NN Num Acc",
        "Perceptron Face Acc",
        "Manual NN Face Acc",
        "PyTorch NN Face Acc", 
        ]
        
    ]

    print(tabulate(data))

    #reading all files

    face_trainingimages = file_reader.faceReader("facedatatrain")
    face_traininglabels = file_reader.face_labelReader("facedatatrainlabels")

    face_testimages = file_reader.faceReader("facedatatest")
    face_testlabels = file_reader.face_labelReader("facedatatestlabels")

    num_trainingimages = file_reader.numberReader("trainingimages")
    num_traininglabels = file_reader.number_labelReader("traininglabels")

    num_testimages = file_reader.numberReader("testimages")
    num_testlabels = file_reader.number_labelReader("testlabels")

    percent = 0.1

    for i in range(10): #in 10 percent increment
        tests = 5 #for each model

        time_perceptron_num = [0] * tests
        time_manal_nn_num = [0] * tests
        time_pytorch_nn_num = [0] * tests
        time_perceptron_face = [0] * tests
        time_manal_nn_face = [0] * tests
        time_pytorch_nn_face = [0] * tests

        accuracy_perceptron_num = [0] * tests
        accuracy_manual_nn_num = [0] * tests
        accuracy_pytorch_nn_num = [0] * tests
        accuracy_perceptron_face = [0] * tests
        accuracy_manual_nn_face = [0] * tests
        accuracy_pytorch_nn_face = [0] * tests

        print("current percent : ", percent*100, "%")
         
        for t in range(tests):

            print("current test : ", t)

            face_train_combined = list(zip(face_trainingimages, face_traininglabels))
            num_train_combined = list(zip(num_trainingimages, num_traininglabels))


            random.shuffle(face_train_combined)
            random.shuffle(num_train_combined)

            face_trainingimages, face_traininglabels = zip(*face_train_combined)
            num_trainingimages, num_traininglabels = zip(*num_train_combined)

            face_trainingimages = list(face_trainingimages)
            face_traininglabels = list(face_traininglabels)
            num_trainingimages = list(num_trainingimages)
            num_traininglabels = list(num_traininglabels)

            new_face_trainingimages = face_trainingimages[:int(percent * len(face_trainingimages))]
            new_face_traininglabels = face_traininglabels[:int(percent * len(face_traininglabels))]

            new_num_trainingimages = num_trainingimages[:int(percent * len(num_trainingimages))]
            new_num_traininglabels = num_traininglabels[:int(percent * len(num_traininglabels))]


            

            print("some lengths : ", len(new_face_trainingimages), len(face_testimages), len(new_num_trainingimages), len(num_testimages))

            #print(new_face_trainingimages)

            #face perceptron
            start_time = time.time()
            accuracy_perceptron_face[t] = face_perceptron_runner(new_face_trainingimages, new_face_traininglabels, face_testimages, face_testlabels)
            end_time = time.time()

            time_perceptron_face[t] = end_time - start_time

            #num perceptron
            start_time = time.time()
            accuracy_perceptron_num[t] = num_perceptron_runner(new_num_trainingimages, new_num_traininglabels, num_testimages, num_testlabels)
            end_time = time.time()

            time_perceptron_num[t] = end_time - start_time

            #face manual nn
            start_time = time.time()
            accuracy_pytorch_nn_face[t] = face_manual_NN_runner(new_face_trainingimages, new_face_traininglabels, face_testimages, face_testlabels)
            end_time = time.time()

            time_manal_nn_face[t] = end_time - start_time
            
            #num manual nn
            start_time = time.time()
            accuracy_pytorch_nn_num[t] = num_manual_NN_runner(new_num_trainingimages, new_num_traininglabels, num_testimages, num_testlabels)
            end_time = time.time()

            time_manal_nn_num[t] = end_time - start_time
            
            #face pytorch nn
            start_time = time.time()
            accuracy_manual_nn_face[t] = face_pytorch_NN_runner(new_face_trainingimages, new_face_traininglabels, face_testimages, face_testlabels)
            end_time = time.time()

            time_pytorch_nn_face[t] = end_time - start_time
            
            #num pytorch nn
            start_time = time.time()
            accuracy_manual_nn_num[t] = num_pytorch_NN_runner(new_num_trainingimages, new_num_traininglabels, num_testimages, num_testlabels)
            end_time = time.time()

            time_pytorch_nn_num[t] = end_time - start_time

            #notes : randomly sort the data for each of the percents so we can send it to the 
        
        data.append([percent * 100, uncertainty(accuracy_perceptron_face), uncertainty(accuracy_perceptron_num), uncertainty(accuracy_manual_nn_face), uncertainty(accuracy_manual_nn_num), uncertainty(accuracy_pytorch_nn_face), uncertainty(accuracy_pytorch_nn_num)])
        print(tabulate(data))
        data.append([percent * 100, uncertainty(time_perceptron_face), uncertainty(time_perceptron_num), uncertainty(time_manal_nn_face), uncertainty(time_manal_nn_num), uncertainty(time_pytorch_nn_face), uncertainty(time_pytorch_nn_num)])
        print(tabulate(data))
        percent = round((percent + 0.1), 2)

    print(tabulate(data))

        


        
