# gather and First Glance at the Data
import pandas as pd

# read the CSV file
data = pd.read_csv('../data/creditcard.csv')
print(data.describe())

# only use the 'Amount' and 'V1', ..., 'V28' features
features = ['Amount'] + ['V%d' % number for number in range(1, 29)]

# the target variable which we would like to predict, is the 'Class' variable
target = 'Class'

# now create an X variable (containing the features) and an y variable (containing only the target variable)
X = data[features]
Y = data[target]

# include all the dependendies
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt

# define the model
model = Sequential()
model.add(Dense(units=58,
    input_dim=29,
    activation='relu'))
model.add(Dense(units=40,
    activation='relu'))
model.add(Dense(units=15,
    activation='relu'))

# output layer
model.add(Dense(units=1,
    activation='sigmoid'))

# load pre-trained model weights
model.load_weights('credit_model.h5')

# compile
model.compile(loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# training model
history = model.fit(X, Y,
    validation_split=0.20,
    epochs=10,
    batch_size=32)

model.save_weights('credit_model.h5')

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()