import file_reader
import random

def dot_product(weights, x, bias):
    total = 0.0

    #print(weights)

    for i in range(len(weights)):
        total += (weights[i] * x[i])

    total += bias

    return total

def evaluate(images, labels, new_weights, new_bias):
    correct = 0

    for i in range(len(labels)):
        x = images[i]
        y = labels[i]
        y_predict = -1

        #print(len(new_weights))

        scores = []

        for j in range(2):
                score = dot_product(new_weights[j], x, new_bias[j])
                scores.append(score)
        
        y_predict = scores.index(max(scores))

        if y_predict == y:
            correct += 1
        
    print(correct/len(images))


def perceptron(images, labels, weights, biases, learning_rate):
    for epoch in range(10):

        print(epoch)

        for i in range(len(labels)):
            x = images[i]
            y_real = labels[i]
            y_predict = -1

            scores = [] #score for each of the values from 0-9

            for j in range(2):
                score = dot_product(weights[j], x, biases[j])
                scores.append(score)
            
            y_predict = scores.index(max(scores))

            # for j in range(10):
            #     if scores[j] > y_predict:
            #         y_predict = j

            y_predict = scores.index(max(scores))


            #print(y_predict)
                

            if y_predict != y_real:

                for j in range(len(x)):
                    weights[y_real][j] += (learning_rate * x[j])
                    weights[y_predict][j] -= (learning_rate * x[j])
                
                biases[y_real] += learning_rate
                biases[y_predict] -= learning_rate

    return weights, biases


if __name__ == "__main__":
    print("main in perceptron")

    trainingimages = file_reader.faceReader("facedatatrain")
    traininglabels = file_reader.face_labelReader("facedatatrainlabels")

    testimages = file_reader.faceReader("facedatatest")
    testlabels = file_reader.face_labelReader("facedatatestlabels")

    num_classes=2 #tells if face or not, highest prob is the output.
    learning_rate = 0.1
    input_size = len(trainingimages[0])
    print(input_size)

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


    new_weights, new_bias = perceptron(trainingimages, traininglabels, weights, biases, learning_rate)

    evaluate(testimages, testlabels, new_weights, new_bias)

    