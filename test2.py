import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
mnist = tf.keras.datasets.mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()
X_train = X_train/255.0
X_test = X_test/255.0


model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1:]),activation='relu',return_sequences = True))#return_sequences set to true implies that the network returns or outputs sequential data else flat data, if next layer is an RNN, set return_sequential to True.
model.add(Dropout(0.2))
model.add(LSTM(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation = 'softmax'))

opt = tf.keras.optimizers.Adam(lr =1e-3, decay=1e-5) #lr is learning rate, and decay implies that learning rate gradually decreases so that we take smaller steps as time progresses where we could be around a local minima, we wouldn't want to jump around local minima but to go right into it
model.compile(loss='sparse_categorical_crossentropy',optimizer = opt,metrics=['accuracy'])

model.fit(X_train,y_train,epochs=3,validation_data = (X_test,y_test))