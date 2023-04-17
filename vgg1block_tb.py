import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np
from random import seed
import time
import datetime
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
seed(1)

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

def log_images_predictions(writer, images, labels, preds_classes, step):
    # convert predictions from probabilities to class labels
    preds_classes = [1 if x > 0.5 else 0 for x in preds_classes]
    # log test images and predictions on tensorboard
    with writer.as_default():
        tf.summary.image("Test Images", images, max_outputs=len(images), step=step)
        tf.summary.text("True Labels", np.array2string(labels, separator=', '), step=step)
        tf.summary.text("Predicted Labels", np.array2string(preds_classes, separator=', '), step=step)
    writer.flush()
    
# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	# pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	pyplot.legend()
	# plot accuracy
	pyplot.subplot(212)
	# pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.legend()
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# run the test harness for evaluating a model
def run_test_harness():
    # define model
    model = define_model()
    # logs="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)

    # callback to log losses
    loss_callback = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: tf.summary.scalar('loss', logs['loss'], epoch))
    # create data generator
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    # prepare iterators
    train_it = datagen.flow_from_directory('dataset_jackel_vs_nilgai/train/',
        class_mode='binary', batch_size=64, target_size=(200, 200))
    # print(train_it[0])
    test_it = datagen.flow_from_directory('dataset_jackel_vs_nilgai/test/',
        class_mode='binary', batch_size=64, target_size=(200, 200))

    # fit model
    start_time = time.time()
    model.fit(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=2, callbacks=[tensorboard_callback])

    end_time = time.time()
    print("Time taken to train the model: ", end_time - start_time, "s")

    print("Training loss: ", model.history.history['loss'][-1])
    print("Training accuracy: ", (model.history.history['accuracy'][-1])*100.0)
    # evaluate model
	# _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    # print('Testing Accuracy: ', (acc * 100.0))

    print("Number of parameters: ", model.count_params())

    # learning curves
    summarize_diagnostics(model.history)


# entry point, run the test harness
run_test_harness()
