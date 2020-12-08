import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from sklearn.utils import class_weight
from data_preprocessing import gendata

def get_dataset(doPCA = False):
	train_X, test_X, train_Y, test_Y = gendata(doPCA)
	train_Y = train_Y - 1
	test_Y = test_Y - 1
	output_size = train_Y.max() - train_Y.min()+1
	return train_X, test_X, train_Y, test_Y, output_size

#It needs at least tensorflow(1.14) and Sklearn(0.22).
def tina(train_X, test_X, train_Y, test_Y, output_size, class_weights):
        model = tf.keras.Sequential()
        #34input, 512hidden
        model.add(layers.Dense(512, input_shape=(34,), activation='relu'))
        model.add(layers.Dropout(0.5))
        #512input,1024 hidden
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        #32*32=1024
        model.add(tf.keras.layers.Reshape((32, 32, 1)))
        #64 kernel，64 feature map
        model.add(layers.Conv2D(64, 3, activation='relu'))
        model.add(layers.Conv2D(32, 3, activation='relu'))
        model.add(tf.keras.layers.Flatten())
        #17 output
        model.add(layers.Dense(output_size, kernel_initializer='glorot_uniform'))
        model.summary()

        lr=0.01
        epoch = 10000
        sgd = optimizers.SGD(lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss = keras.losses.SparseCategoricalCrossentropy
                      (from_logits=True), optimizer=sgd, metrics=['accuracy'])

        # here we use the test_set as the validation_set    
        model.fit(train_X,train_Y,batch_size=64,epochs=epoch,verbose=2,
                  validation_data=(test_X,test_Y), class_weight=class_weights)
        score = model.evaluate(test_X,test_Y,verbose=0)
        print(score)

def main():
        train_X, test_X, train_Y, test_Y, output_size = get_dataset()
        class_weights = class_weight.compute_class_weight('balanced',
                                                    np.unique(train_Y),
                                                    train_Y)
        tina(train_X, test_X, train_Y, test_Y, output_size, class_weights)
main()



