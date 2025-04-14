import os


def numberReader(image_file):
    #print("image")
    
    image_file = "digitdata\\" + image_file

    with open(image_file, "r") as file:
        lines = file.readlines()

    image_numbers = []
    current = []
    count = 0

    max = len(lines)
    #max = 50

    while count < max:
        #print(lines[count])
        while count < max and lines[count].strip() == '':
            #print("empty line", lines[count])
            count += 1
        
        current = []

        if count + 20 > max:
            break


        for i in range(20):
            #print(i, " ", count, lines[count])
            
            current.append(lines[count].rstrip('\n'))
            count += 1

        if current:
            image_numbers.append(current)
    
    print("numbers of images this data set :", len(image_numbers))

    return image_numbers


def number_labelReader(label_file):
    #print("label")

    label_file = "digitdata\\" + label_file

    with open(label_file, "r") as file:
        lines = file.readlines()

    labels = []

    for line in lines:
        labels.append(int(line))

    print("numbers of images this data set :", len(labels))

def faceReader(image_file):
    #print("image")
    
    image_file = "facedata\\" + image_file

    with open(image_file, "r") as file:
        lines = file.readlines()

    image_numbers = []
    current = []
    count = 0

    max = len(lines)
    #max = 50

    while count < max:
        
        current = []

        if count + 70 > max:
            break

        for i in range(70):
            #print(i, " ", count, lines[count])
            
            current.append(lines[count].rstrip('\n'))
            count += 1

        if current:
            image_numbers.append(current)
    
    print("numbers of faces this data set :", len(image_numbers))

    return image_numbers

def face_labelReader(label_file):
    #print("label")

    label_file = "facedata\\" + label_file

    with open(label_file, "r") as file:
        lines = file.readlines()

    labels = []

    for line in lines:
        labels.append(int(line))

    print("numbers of faces this data set :", len(labels))
    
    


if __name__ == "__main__":
    print("main in image reader")

    image_file = "testimages"
    numberReader(image_file)

    label_file = "testlabels"
    number_labelReader(label_file)

    label_file = "facedatatest"
    faceReader(label_file)

    label_file = "facedatatestlabels"
    face_labelReader(label_file)

    
        