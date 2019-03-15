import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
import h5py
import pickle
import time

dense_layers = [0]
layer_sizes = [128]
conv_layers = [3]

BATCH_SIZE = 32
EPOCHS = 25
VALIDATION_SPLIT = 0.5

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

'''train_X = pickle.load(open("train_X.pickle", "rb"))
train_y = pickle.load(open("train_y.pickle", "rb"))
test_X = pickle.load(open("test_X.pickle", "rb"))
test_y = pickle.load(open("test_y.pickle", "rb"))'''
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0
#test_X - test_X/255.0

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = f"Cats_vs_dogs-{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{int(time.time())}"

            model = Sequential()

            model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())
            model.add(Dropout(0.2))
            
            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1, activation="softmax"))

            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

            opt = tf.keras.optimizers.Adam(lr=0.01, decay=1e-6)

            model.compile(loss="binary_crossentropy",
                          optimizer=opt,
                          metrics=['accuracy'])

            model.fit(X,y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT, callbacks=[tensorboard], shuffle=True)

            model.save(NAME)
