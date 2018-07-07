# Image Classification

# Import libraries
from keras import backend
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization 
from keras.regularizers import l2 # L2-regularisation


# dimensions of our images.
imw, imh = 64, 64

if backend.image_data_format() == 'channels_first':
    ins = (3, imw, imh)
else:
    ins = (imw, imh, 3)

# Initalize CNN
model = Sequential()

# Add 2 convolution layers --- Conv [32] -> Conv [32] -> Pool
model.add(Conv2D(filters=32,
    kernel_size=(3, 3),
    input_shape=ins,
    padding='same',
    kernel_initializer='he_uniform',
    kernel_regularizer=l2(0.0001),
    activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32,
    kernel_size=(3, 3),
    padding='same',
    kernel_initializer='he_uniform',
    kernel_regularizer=l2(0.0001),
    activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Add 2 more convolution layers --- Conv [64] -> Conv [64] -> Pool
model.add(Conv2D(filters=64,
    kernel_size=(3, 3),
    padding='same',
    kernel_initializer='he_uniform',
    kernel_regularizer=l2(0.0001),
    activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64,
    kernel_size=(3, 3),
    padding='same',
    kernel_initializer='he_uniform',
    kernel_regularizer=l2(0.0001),
    activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Add 2 more convolution layers --- Conv [128] -> Conv [128] -> Pool (with dropout on the pooling layer)
model.add(Conv2D(filters=128,
    kernel_size=(3, 3),
    padding='same',
    kernel_initializer='he_uniform',
    kernel_regularizer=l2(0.0001),
    activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=128,
    kernel_size=(3, 3),
    padding='same',
    kernel_initializer='he_uniform',
    kernel_regularizer=l2(0.0001),
    activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

# Add full connection --- Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
model.add(Flatten())
model.add(Dense(units=128,
    kernel_initializer='he_uniform',
    kernel_regularizer=l2(0.0001),
    activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(units=2,
    kernel_initializer='glorot_uniform',
    kernel_regularizer=l2(0.0001),
    activation='softmax'))

# Load model weight
model.load_weights('weight_model.h5')

# Compiling the ANN
model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Fit CNN to images
from keras.preprocessing.image import ImageDataGenerator

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

test_datagen = ImageDataGenerator()

train_data = train_datagen.flow_from_directory('dataset/face/train',
    target_size=(imw, imh),
    batch_size=32,
    class_mode='categorical')

test_data = test_datagen.flow_from_directory('dataset/face/test',
    target_size=(imw, imh),
    batch_size=32,
    class_mode='categorical')

model.fit_generator(train_data,
    steps_per_epoch=len(train_data),
    epochs=10,
    workers=4,
    shuffle=True,
    validation_data=test_data,
    validation_steps=len(test_data))

# model.fit_generator(train_data,
    # steps_per_epoch=len(train_data),
    # epochs=50,
    # shuffle=True,
    # workers=4)

model.save('cnn_model.h5')
model.save_weights('weight_model.h5')