import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# defining MLP model
def define_model():
    model = Sequential()
    model.add(Flatten(input_shape=(200, 200, 3)))
    model.add(Dense(2048, activation='relu'))           # 123 million parameters
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))           # add the output layer
    
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# plotting diagnostic learning curves
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
    model = define_model()                            # define model        
    datagen = ImageDataGenerator(rescale=1.0/255.0)   # create data generator
    train_it = datagen.flow_from_directory('dataset_jackel_vs_nilgai/train/',
                                            class_mode='binary', batch_size=64, target_size=(200, 200))     # prepare iterators
    test_it = datagen.flow_from_directory('dataset_jackel_vs_nilgai/test/',
                                            class_mode='binary', batch_size=64, target_size=(200, 200))     # prepare iterators
    history = model.fit(train_it, steps_per_epoch=len(train_it),
                        validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)       # fit model

    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)                                         # evaluate model
    print('> %.3f' % (acc * 100.0))
    summarize_diagnostics(history)                                                                          # learning curves

# entry point, run the test harness
run_test_harness()
