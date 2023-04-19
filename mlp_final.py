import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from random import seed
import time
import datetime
import os
import matplotlib.pyplot as plt
import io
import keras
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
seed(1)

# define MLP model
def define_model():
    model = Sequential()
    model.add(Flatten(input_shape=(200, 200, 3)))
    model.add(Dense(1024, activation='relu')) ## 240 million parameters
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))

# Add the output layer
    model.add(Dense(1, activation='sigmoid'))
    
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    print("Number of parameters: ", model.count_params())
    return model

def plot_to_image(figure):
    """Converts a Matplotlib figure to a TensorFlow image tensor."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()

# run the test harness for evaluating a model
def run_test_harness():
    # define model
    model = define_model()
    logs="logs/fit/mlp/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logs, histogram_freq=1)
    # create data generator
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    # prepare iterators
    train_it = datagen.flow_from_directory('dataset_jackel_vs_nilgai/train/',
                                            class_mode='binary', batch_size=40, target_size=(200, 200))
    test_it = datagen.flow_from_directory('dataset_jackel_vs_nilgai/test/',
                                            class_mode='binary', batch_size=40, target_size=(200, 200))
    # fit model
    start_time = time.time()
    history = model.fit(train_it, steps_per_epoch=len(train_it),
        validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=2, callbacks = [tensorboard_callback])
    end_time = time.time()
    print("Time taken to train the model: ", end_time - start_time, "s")
    print("Training Accuracy: ", (history.history['accuracy'][-1])*100.0)
    print("Training loss: ", history.history['loss'][-1])
    # evaluate model
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    print('Testing Accuracy', (acc * 100.0))
    test_it.reset()       
    test_images, test_labels = next(test_it)   
    test_preds = model.predict(test_images, verbose=0) 
    print(len(test_preds))
    # convert predictions from probabilities to class labels  
    test_preds_classes = [0 if x>0.5 else 1 for x in test_preds]
    # log test images and predictions on tensorboard
    logdir = "logs/mlp/"   	
    
    file_writer = tf.summary.create_file_writer(logdir)   
    with file_writer.as_default():
        figure, axes = plt.subplots(nrows=4, ncols=8, figsize=(15, 10))        
        for i, ax in enumerate(axes.flat):           
            if i < len(test_images):
                ax.imshow(test_images[i])
                ax.set_title("True: {} Pred: {}".format(test_labels[i], test_preds_classes[i]))
                ax.axis("off")
        
        plt.tight_layout()
        tf.summary.image("Test Images", plot_to_image(figure), step=0)		
        count = 0
        for i in range(len(history.history['accuracy'])):
            tf.summary.scalar("accuracy/training/final", history.history['accuracy'][i], step=(i+1)*4)
            tf.summary.scalar("loss/training/final", history.history['loss'][i], step=(i+1)*4)
            tf.summary.scalar("accuracy/testing/final", history.history['val_accuracy'][i], step=i)
            tf.summary.scalar("loss/testing/final", history.history['val_loss'][i], step=i)
    print("Number of parameters: ", model.count_params())
    # learning curves
    summarize_diagnostics(history)

# entry point, run the test harness
run_test_harness()
