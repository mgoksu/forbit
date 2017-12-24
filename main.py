
import csv

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam
import matplotlib.pyplot as plt


pred_ahead = 1
last_n = 5
train_size = 1000
val_size = 100
test_size = 200
batch_size = 16
epochs = 200


def create_model():
    mdl = Sequential()
    mdl.add(Dense(1024, activation='relu', input_shape=(last_n,)))
    mdl.add(Dense(1024, activation='relu'))
    mdl.add(Dense(1024, activation='relu'))
    mdl.add(Dense(1024, activation='relu'))
    mdl.add(Dense(1))

    return mdl


prices = []
# read csv
with open('market-price.csv', newline='') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    for rw in csv_reader:
        prices.append(float(rw[0].split(',')[1]))

# omit zero prices at the beginning
prices = [i for i in prices if i != 0]

# change in prices as percentage
change_arr = np.zeros((len(prices) - 1, 1))
for i in range(1, len(prices)):
    change_arr[i-1, 0] = (prices[i] - prices[i-1]) / prices[i]

y_change = change_arr[last_n + pred_ahead - 1:, 0]
x_change = np.zeros((len(y_change), last_n))
for i in range(len(x_change)):
    x_change[i, :] = change_arr[i: i + last_n, 0]


train_x = x_change[:train_size]
train_y = y_change[:train_size]

val_x = x_change[train_size:train_size+val_size]
val_y = y_change[train_size:train_size+val_size]

test_x = x_change[train_size+val_size:]
test_y = y_change[train_size+val_size:]

model = create_model()

model.summary()
model.compile(loss='mse', optimizer=adam())

model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(val_x, val_y))

y_pred = model.predict(test_x, batch_size=len(test_y))
errs = model.evaluate(test_x, test_y)

print('MSE Error: {0:.4f}'.format(errs))
t = np.arange(0, len(y_pred))
plt.plot(t, test_y, 'g', t, y_pred, 'b')
plt.legend(['Real Change', 'Estimation'])
plt.show()
