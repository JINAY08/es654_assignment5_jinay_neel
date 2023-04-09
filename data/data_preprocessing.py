
# load jackels vs nilgai dataset, reshape and save to a new file
from os import listdir, makedirs
from numpy import asarray
from numpy import save
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import PIL
from shutil import copyfile, copy
from random import seed
from random import random
# define location of dataset
folder = 'data/images/'
photos, labels = list(), list()
# enumerate files in the directory
i = 0
for file in listdir(folder):
	# determine class
        photo = load_img(folder + file, target_size=(200, 200))
        output = 0.0
        if file.startswith('jackal'):
            output = 1.0
	    # load image
	# convert to numpy array
        photo = img_to_array(photo)
	# store
        photos.append(photo)
        labels.append(output)

# convert to a numpy arrays
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)
# save the reshaped photos
save('jackel_vs_nilgai_photos.npy', photos)
save('jackel_vs_nilgai_labels.npy', labels)

# create directories
dataset_home = 'dataset_jackel_vs_nilgai/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
	# create label subdirectories
	labeldirs = ['jackel/', 'nilgai/']
	for labldir in labeldirs:
            newdir = dataset_home + subdir + labldir
            makedirs(newdir, exist_ok=True)

# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.2
# copy training dataset images into subdirectories
src_directory = 'data/images'
for file in listdir(src_directory):
	src = src_directory + "/" + file
	print(src)
	dst_dir = 'train/'
	if random() < val_ratio:
		dst_dir = 'test/'
	if file.startswith('jackal'):
		dst = dataset_home + dst_dir + "jackel/" + file
	elif file.startswith('nilgai'):
		dst = dataset_home + dst_dir + 'nilgai/' + file
	copyfile(src, dst)
	


