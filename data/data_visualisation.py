import matplotlib.pyplot as plt
import os

# Path: data\data_visualisation.py

def show_images():
    path = "data/images/" 
    i = 0
    for filename in os.listdir(path):
        if(filename.startswith("jackal")):
            plt.subplot(330 + 1 + i)
            img = imread(path + filename)      
            plt.imshow(img)
            i += 1
            if i == 9:
                break
    plt.show()

show_images()
