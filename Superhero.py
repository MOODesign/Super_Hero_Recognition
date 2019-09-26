import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

train_dir = os.path.join("D:\Data_super_hero\Superhero_Train") #the directory where the training images are saved.
CATEGORIES = ["AntMan", "Aquaman", "Avengers", "Batman", "Black Panther", "Captain America", "Catwoman", "Ghost Rider", "Hulk", "Iron Man", "Spiderman", "Superman"]
# we will be classifying our images into 12 categories for 12 superhers , there will be a graph to help with the identification.
for category in CATEGORIES:
    path = os.path.join(train_dir, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #turning all images gray scale so that the training can run faster and since we don't need color to identify features.
        plt.imshow(img_array, cmap="gray")
        #plt.show()
        break
    break

new_array = cv2.resize(img_array, (50, 50)) #all images need to be resize to the same size, i choose 50x50 as this won't be so blurry yet won't kill my computer while training.
#plt.imshow(new_array, cmap="gray")
#plt.show()

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(train_dir, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try: # this exception is for when there's a damaged image the script will skip it.
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (50, 50))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
#print(len(training_data)) #uncomment this to see the amount of training data.

import random
random.shuffle(training_data) # shuffling the images is useful so that the model won't learn form a specific category first then a different one.


x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, 50, 50, 1) # a needed step to reshape all data again into 50x50 pixels and in gray scale, because keras doesnt support it automatically yet.

#saving the x features as an array in a binary file in .npy format.
np.save('features.npy',x)
x = np.load('features.npy')

