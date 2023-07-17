import pandas as pd
import numpy as np
import pickle

data=pd.read_csv('C:/Users/Dell/Downloads/MiniDatatset.csv')

x = data.iloc[:, [1, 2, 4, 5, 6, 7, 8, 11, 14, 16, 17, 18, 19]]
y = data.iloc[:, 0]

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

from tensorflow import keras

model = keras.models.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(13,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(xtrain, ytrain, epochs=10, batch_size=32)

_, accuracy = model.evaluate(xtest, ytest)
print("Accuracy of model through ANN classifier is:", accuracy * 100)

# Save the model to a file
#with open('model.sav', 'wb') as f:
#    pickle.dump(model, f)
model.save("model.h5")
