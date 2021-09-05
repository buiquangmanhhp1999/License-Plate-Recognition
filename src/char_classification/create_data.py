import os
import numpy as np
import cv2

path = "./data/categorized/digits/"
data = []


for fi in os.listdir(path):
    if fi == "0":
        label = 21
    elif fi == "1":
        label = 22
    elif fi == "2":
        label = 23
    elif fi == "3":
        label = 24
    elif fi == "4":
        label = 25
    elif fi == "5":
        label = 26
    elif fi == "6":
        label = 27
    elif fi == "7":
        label = 28
    elif fi == "8":
        label = 29
    elif fi == "9":
        label = 30
    elif fi == "BG":
        label = 31
    else:
        label = -1
        ValueError("Don't match file")

    img_fi_path = os.listdir(path + fi)
    for img_path in img_fi_path:
        img = cv2.imread(path + fi + "/" + img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
        img = img.reshape((28, 28, 1))
        data.append((img, label))


np.save("./data/digits.npy", data)


path = "./data/categorized/alphas/"
data = []


for fi in os.listdir(path):
    if fi == "A":
        label = 0
    elif fi == "B":
        label = 1
    elif fi == "C":
        label = 2
    elif fi == "D":
        label = 3
    elif fi == "E":
        label = 4
    elif fi == "F":
        label = 5
    elif fi == "G":
        label = 6
    elif fi == "H":
        label = 7
    elif fi == "K":
        label = 8
    elif fi == "L":
        label = 9
    elif fi == "M":
        label = 10
    elif fi == "N":
        label = 11
    elif fi == "P":
        label = 12
    elif fi == "R":
        label = 13
    elif fi == "S":
        label = 14
    elif fi == "T":
        label = 15
    elif fi == "U":
        label = 16
    elif fi == "V":
        label = 17
    elif fi == "X":
        label = 18
    elif fi == "Y":
        label = 19
    elif fi == "Z":
        label = 20
    else:
        label = -1
        ValueError("Don't match file")

    img_fi_path = os.listdir(path + fi)
    for img_path in img_fi_path:
        img = cv2.imread(path + fi + "/" + img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
        img = img.reshape((28, 28, 1))
        data.append((img, label))


np.save("./data/alphas.npy", data)

