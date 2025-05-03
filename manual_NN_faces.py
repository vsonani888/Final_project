import file_reader
import random
import math
import json

def dot_product(weights, x, bias):
    total = 0.0

    #print(len(weights), len(x))

    for i in range(len(weights)):
        total += (weights[i] * x[i])

    total += bias

    return total

def sigmoid(z):

    result = []

    for i in range(len(z)):
        result.append(1/(1 + math.exp(-z[i])))

    return result

def softmax(z):
    # exp = []
    # exp_total = 0

    # for i in range(len(z)):
    #     exp.append(math.exp(z[i]))
    #     exp_total += exp[i]
    
    # softed = []

    # for i in range(len(z)):
    #     softed.append(exp[i] / exp_total)

    m = max(z)
    exp = [math.exp(zi-m)     for zi in z]
    total = sum(exp)
    softed = [e/total for e in exp]


    return softed

# def sigmoid_derivative(z):
    
#     sigmoidted_z = []
#     result = []

#     for i in range(len(z)):
#         sigmoidted_z.append(1/(1 + math.exp(-z[i])))
    
#     for i in range(len(z)):
#         result = sigmoidted_z[i] * (1-sigmoidted_z[i])

#     return result

def sigmoid_derivative(z):

    s = 1 / (1 + math.exp(-z))
    result = s * (1 - s)

    return result 

def fill(input_size, output_size, hidden_1_size, hidden_2_size, weights_1, weights_2, weights_3, biases_1, biases_2, biases_3):
    for i in range(hidden_1_size):

        bias1 = random.uniform(-0.01, 0.01)
        biases_1.append(bias1)

        weight_row_1 = []

        for j in range(input_size):
            
            weight_1 = random.uniform(-0.01, 0.01)
            weight_row_1.append(weight_1)
        
        weights_1.append(weight_row_1)

    for i in range(hidden_2_size):

        bias2 = random.uniform(-0.01, 0.01)
        biases_2.append(bias2)

        weight_row_2 = []

        for j in range(hidden_1_size):

            weight_2 = random.uniform(-0.01, 0.01)
            weight_row_2.append(weight_2)
        
        weights_2.append(weight_row_2)

    for i in range(output_size):

        bias3 = random.uniform(-0.01, 0.01)
        biases_3.append(bias3)

        weight_row_3 = []

        for j in range(hidden_2_size):

            weight_3 = random.uniform(-0.01, 0.01)
            weight_row_3.append(weight_3)
        
        weights_3.append(weight_row_3)

    print("size of weights 1", len(weights_1), len(weights_1[0]))
    print("size of weights 2", len(weights_2), len(weights_2[0]))
    print("size of weights 3", len(weights_3), len(weights_3[0]))

    print("size of biases 1", len(biases_1))
    print("size of biases 2", len(biases_2))
    print("size of biases 3", len(biases_3))

    return weights_1, weights_2, weights_3, biases_1, biases_2, biases_3

def training(trainingimages, traininglabels,epochs, learning_rate, output_size, hidden_1_size, hidden_2_size, weights_1, weights_2, weights_3, biases_1, biases_2, biases_3):

    for epoch in range(epochs):

        print("epoch = ", epoch)
        
        
        for i in range(len(trainingimages)): #goes over every image

            #print("training image = ", i)

            x = trainingimages[i]
            y = traininglabels[i]

            y_bin_list = []


            for j in range(output_size):
                if y == j:
                    y_bin_list.append(1)
                else:
                    y_bin_list.append(0)

            #forward pass

            Z1 = []
            Z2 = []
            Z3 = []

            A1 = []
            A2 = []
            A3 = []

            for j in range(hidden_1_size): #goes over first and second hidden layer size 560
                
                Z1.append(dot_product(weights_1[j], x, biases_1[j]))
            
            A1 = sigmoid(Z1)
        
            for j in range(hidden_2_size): #goes over first and second hidden layer size 560

                Z2.append(dot_product(weights_2[j], A1, biases_2[j]))
            
            A2 = (sigmoid(Z2))

            for j in range(output_size): #goes over last layer size 10

                Z3.append(dot_product(weights_3[j], A2, biases_3[j]))
            
            A3 = (softmax(Z3))

            #backward pass

            #last layer
            LZ3 = []
            LW3 = []
            LB3 = []

            #second hidden layer
            LA2 = []
            LZ2 = []
            LW2 = []
            LB2 = []

            #first hidden layer
            LA1 = []
            LZ1 = []
            LW1 = []
            LB1 = []

            #print(A3)

            for j in range(output_size): #goes over last layer size 10
                
                LZ = A3[j] - y_bin_list[j] #error for output node j
                LZ3.append(LZ) #diff between our value and actual value

                LW3_single = []
                for k in range(len(A2)): #for each node in the hidden layer
                    LW3_single.append(LZ * A2[k]) #weight error from hidden 2 to output layer
                LW3.append(LW3_single)

                LB3.append(LZ) #bias for the output layer

            #print(weights_3)

            for j in range(len(A2)): #goes over first or second hidden layer size 560
                
                LA2 = 0
                for k in range(output_size): #for each node in the output layer
                    LA2 += weights_3[k][j] * LZ3[k] #sum of all errors
                #LW2.append(LA2) #error in A2
                
                LZ2_single = LA2 * sigmoid_derivative(Z2[j]) #reverse sigmoid function
                LZ2.append(LZ2_single)
                
                LW2_single = []
                for k in range(len(A1)): #for each node in the hidden layer for weight
                    LW2_single.append(LZ2_single * A1[k]) #weight error from hidden 1 to hidden 2
                LW2.append(LW2_single)
                
                LB2.append(LZ2_single) #bias gradient

            for j in range(len(A1)): #goes over first or second hidden layer size 560

                LA1 = 0
                for k in range(len(A2)): #for each node in the hidden layer
                    LA1 += weights_2[k][j] * LZ2[k] #sum of all errors
                #LW1.append(LA1) #error in A1

                LZ1_single = LA1 * sigmoid_derivative(Z1[j]) #reverse sigmoid function
                LZ1.append(LZ1_single) 

                LW1_single = []
                for k in range(len(x)): #for each node in the hidden layer for weight
                    LW1_single.append(LZ1_single * x[k]) #weight error from input layer to hidden 1
                LW1.append(LW1_single)
                
                LB1.append(LZ1_single) #bias gradient
            
            #Gradient Descent Update

            for j in range(len(weights_3)):
                for k in range(len(weights_3[j])):
                    weights_3[j][k] = weights_3[j][k] - (learning_rate * LW3[j][k]) #updates weights
                
                biases_3[j] = biases_3[j] - (learning_rate * LB3[j]) #updates bias
            
            
            for j in range(len(weights_2)):
                for k in range(len(weights_2[j])):
                    weights_2[j][k] = weights_2[j][k] - (learning_rate * LW2[j][k]) #updates weights

                biases_2[j] = biases_2[j] - (learning_rate * LB2[j]) #updates bias
            

            for j in range(len(weights_1)):
                for k in range(len(weights_1[j])):
                    weights_1[j][k] = weights_1[j][k] - (learning_rate * LW1[j][k]) #updates weights

                biases_1[j] = biases_1[j] - (learning_rate * LB1[j]) #updates bias

    return weights_1, weights_2, weights_3, biases_1, biases_2, biases_3

def save(weights_1, weights_2, weights_3, biases_1, biases_2, biases_3):
    with open('manual_NN_faces_weights_1_testing.txt', 'w') as file:
        json.dump(weights_1, file)

    with open('manual_NN_faces_weights_2_testing.txt', 'w') as file:
        json.dump(weights_2, file)

    with open('manual_NN_faces_weights_3_testing.txt', 'w') as file:
        json.dump(weights_3, file)
    
    with open('manual_NN_faces_biases_1_testing.txt', 'w') as file:
        json.dump(biases_1, file)

    with open('manual_NN_faces_biases_2_testing.txt', 'w') as file:
        json.dump(biases_2, file)
    
    with open('manual_NN_faces_biases_3_testing.txt', 'w') as file:
        json.dump(biases_3, file)

def load(weights_1, weights_2, weights_3, biases_1, biases_2, biases_3):
    with open('manual_NN_faces_weights_1.txt', 'r') as file:
        weights_1 = json.load(file)

    with open('manual_NN_faces_weights_2.txt', 'r') as file:
        weights_2 = json.load(file)

    with open('manual_NN_faces_weights_3.txt', 'r') as file:
        weights_3 = json.load(file)
    
    with open('manual_NN_faces_biases_1.txt', 'r') as file:
        biases_1 = json.load(file)

    with open('manual_NN_faces_biases_2.txt', 'r') as file:
        biases_2 = json.load(file)
    
    with open('manual_NN_faces_biases_3.txt', 'r') as file:
        biases_3 = json.load(file)
    
    return weights_1, weights_2, weights_3, biases_1, biases_2, biases_3

def evalulate(testimages, testlabels, weights_1, weights_2, weights_3, biases_1, biases_2, biases_3, hidden_1_size, hidden_2_size, output_size):
    
    correct = 0
    test_num = len(testimages)
    

    for i in range(test_num): #for each test image

        #print("testing image = ", i)

        x = testimages[i]
        y = testlabels[i] 

        #forward pass

        Z1 = []
        Z2 = []
        Z3 = []

        A1 = []
        A2 = []
        A3 = []

        for j in range(hidden_1_size): #goes over first and second hidden layer size 560
                
            Z1.append(dot_product(weights_1[j], x, biases_1[j]))
            
        A1 = sigmoid(Z1)
        
        for j in range(hidden_2_size): #goes over first and second hidden layer size 560

            Z2.append(dot_product(weights_2[j], A1, biases_2[j]))
        
        A2 = (sigmoid(Z2))

        for j in range(output_size): #goes over last layer size 10

            Z3.append(dot_product(weights_3[j], A2, biases_3[j]))
        
        A3 = (softmax(Z3))
        
        predicted_y = A3.index(max(A3)) #class with highest prob

        if predicted_y == y:
            correct += 1

    print(correct/test_num)

    return correct/test_num



if __name__ == "__main__":
    print("main in manual neural network")

    #reading all files

    trainingimages = file_reader.faceReader("facedatatrain")
    traininglabels = file_reader.face_labelReader("facedatatrainlabels")

    testimages = file_reader.faceReader("facedatatest")
    testlabels = file_reader.face_labelReader("facedatatestlabels")

    #some initialization

    input_size = len(trainingimages[0])

    output_size = 2
    learning_rate = 0.1

    hidden_1_size = 128
    hidden_2_size = 64

    weights_1 = [] #for hidden 1, size 128 * 560
    weights_2 = [] #for hidden 2, size 64 * 128
    weights_3 = [] #for hidden 3, size 2 * 64

    biases_1 = [] #for hidden 1, size 128
    biases_2 = [] #for hidden 2, size 64
    biases_3 = [] #for hidden 3, size 2

    epochs = 30

    train = 1 #we want to train the weights and biases, otherwise it will just use the ones we already have saved.
    
    if train == 0:

        print("train from scratch")

        # #just fills up the bias and the weights with random values

        weights_1, weights_2, weights_3, biases_1, biases_2, biases_3 = fill(input_size, output_size, hidden_1_size, hidden_2_size, weights_1, weights_2, weights_3, biases_1, biases_2, biases_3)

        #training on images

        weights_1, weights_2, weights_3, biases_1, biases_2, biases_3 = training(trainingimages, traininglabels, epochs, learning_rate, output_size, hidden_1_size, hidden_2_size, weights_1, weights_2, weights_3, biases_1, biases_2, biases_3)
        print("training done")

        #saving weights and biases

        save(weights_1, weights_2, weights_3, biases_1, biases_2, biases_3)
        print("saving done")

    else:

        print("use already trained weights and biasses")

        #load pre-trained weights and biases

        weights_1, weights_2, weights_3, biases_1, biases_2, biases_3 = load(weights_1, weights_2, weights_3, biases_1, biases_2, biases_3)
        print("loading done")
        
    #evaluating results

    evalulate(testimages, testlabels, weights_1, weights_2, weights_3, biases_1, biases_2, biases_3, hidden_1_size, hidden_2_size, output_size)






        

            

            


    
