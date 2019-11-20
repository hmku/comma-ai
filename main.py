import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D
from sklearn.metrics import mean_squared_error

# set up model
model = Sequential()
model.add(Conv1D(64, 5, input_shape=(128, 44), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(32, 5, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='mse')

print('done setting up model!')

# get training and validation data
X = np.load('data/train_frames.npy')
y = np.fromfile('data/train.txt', sep=' ')[:-1]
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]
X_validate = X[:len(X)//5]
X_train = X[len(X)//5:]
y_validate = y[:len(X)//5]
y_train = y[len(X)//5:]

print('done loading data!')

model.fit(X_train, y_train, epochs=40, batch_size=32)

print('done fitting model!')
print(model.summary())

y_validate_pred = model.predict(X_validate)
print(f'MSE on validation set: {mean_squared_error(y_validate, y_validate_pred)}')

# get test data
X_test = np.load('data/test_frames.npy')
y_test = model.predict(X_test)
y_test = np.insert(y_test, -1, y_test[-1]) # duplicate last element and use that as speed of last frame
                                           # this is not that good, but also probably not egregiously bad either
np.savetxt('data/test.txt', y_test, delimiter='\n')