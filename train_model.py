from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

# Load data set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# y dataset is like so
'''
airplane    0
automobile  1
bird        2
cat         3
deer        4
dog         5
frog        6
horse       7
ship        8
truck       9
'''

# adjust it so instead it sets all birds to true and everything else to false
# broadcast boolean logic througout dataset
y_train = y_train == 2
y_test = y_test == 2

# prepare x to be normalized (between 0 and 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        padding='same',
        input_shape=(32, 32, 3),
        activation="relu"
    )
)
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# second grouping
model.add(
    Conv2D(64, (3, 3), padding='same', activation="relu")
)
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# Final dense layer
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=['accuracy']
)

model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=200,
    validation_data=(x_test, y_test),
    shuffle=True
)

model.save("bird_model.h5")
