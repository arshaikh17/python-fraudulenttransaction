# gather and First Glance at the Data
import pandas as pd

# read the CSV file
# data = pd.read_csv('../data/creditcard_normal.csv')
data = pd.read_csv('../data/creditcard_fraud.csv')
print(data.describe())

# only use the 'Amount' and 'V1', ..., 'V28' features
features = ['Amount'] + ['V%d' % number for number in range(1, 29)]

# now create an X variable (containing the features) and an y variable (containing only the target variable)
X = data[features]

# include all the dependendies
from keras.layers import Dense
from keras.models import Sequential

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

    # predict model
ynew = model.predict_classes(X)
probs = model.predict_proba(X)

# show the inputs and predicted outputs
for i in range(len(X)):
    if ynew[i] == 0:
        class_label = 'Normal'
        prob = 50 + 100 * (0.5 - probs[i])
    else:
        class_label = 'Fradulent'
        prob = 50 + 100 * (probs[i] - 0.5)

    print("%d --> class=%s, confidence=%.2f%%" % (i, class_label, prob))