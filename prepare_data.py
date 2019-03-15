import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "C:/Users/Brandon/Desktop/Neural Network Test/Cat or Dog/Cat-or-Dog/PetImages"
TRAIN_CATEGORIES = ["Dog", "Cat"]
TEST_CATERGORIES = ["DogTest", "CatTest"]
IMG_SIZE = 50
training_data = []
test_data = []

def create_training_data(): 
    for category in TRAIN_CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = TRAIN_CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

def create_testing_data(): 
    for category in TEST_CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = TEST_CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                test_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
create_testing_data
random.shuffle(training_data)
random.shuffle(test_data)

train_X = []
train_y = []

test_X = []
test_y = []

for features, label in training_data:
    train_X.append(features)
    train_y.append(label)

for feature, label in test_data:
    test_X.append(features)
    test_y.append(label)

train_X = np.array(train_X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_X = np.array(test_X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("train_X.pickle", "wb")
pickle.dump(train_X, pickle_out)
pickle_out.close()

pickle_out = open("train_y.pickle", "wb")
pickle.dump(train_y, pickle_out)
pickle_out.close()

pickle_out = open("test_X.pickle", "wb")
pickle.dump(test_X, pickle_out)
pickle_out.close()

pickle_out = open("test_y.pickle", "wb")
pickle.dump(test_y, pickle_out)
pickle_out.close()

pickle_in = open("train_X.pickle", "rb")
train_X = pickle.load(pickle_in)

print("Done")
