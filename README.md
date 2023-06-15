# cari-in-ml
# Building a Machine Learning Model for Bali Tourism Prediction

In this guide, we will provide a detailed explanation of the steps involved in building a machine learning model for predicting tourism in Bali. The provided source code utilizes various libraries such as Pandas, NumPy, scikit-learn, and TensorFlow.

## 1. Importing Required Libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
```

Begin by importing the necessary libraries required to construct and train the machine learning model.

## 2. Reading the Data

```python
df = pd.read_csv("/content/Copy of Data_Wisata_Bali - Sheet1 (3).csv")
df
```

Next, read the Bali tourism data from a CSV file using the Pandas library. Ensure that the file path corresponds to the correct location of the file you wish to read.


## 3. Data Preprocessing

```python
wilayah = np.unique(df["Wilayah"])
```

Retrieve the unique values from the "Wilayah" column, which will serve as the target variable to be predicted.

```python
label_encoder = LabelEncoder()
wilayahEncoder = label_encoder.fit_transform(wilayah)
print(wilayahEncoder)
```


Use the LabelEncoder from scikit-learn to convert categorical values into numerical values that can be processed by the machine learning model.


```python
df_le = pd.DataFrame()
df_le["Wilayah_Encoded"] = label_encoder.fit_transform(df["Wilayah"])
df_le["Panorama_Encoded"] = label_encoder.fit_transform(df["Panorama"])
df_le["Tipe Rekreasi_Encoded"] = label_encoder.fit_transform(df["Tipe Rekreasi"])
df_le["Target Prediksi_Encoded"] = label_encoder.fit_transform(df["Target Prediksi (y)"])

df_le
```

Create a new DataFrame that contains the encoded columns.

```python
X = df_le.drop("Target Prediksi_Encoded", axis=1)
y = df_le["Target Prediksi_Encoded"]
```

Split the data into features (X) and the target variable (y), which will be used to train the model.

## 4. Build the Model

```python
model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(16, activation='relu'))
model.add(Dense(y.unique().shape[0], activation='softmax'))
```


Construct a neural network model using TensorFlow. The model consists of multiple dense layers with the specified activation functions.

## 5. Compile the Model

```python

lr = 0.015  # Desired learning rate

optimizer = SGD(learning_rate=lr)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

Compile the model using the specified optimizer, loss function, and evaluation metrics.


## 6. Train the Model

```python

history_model = model.fit(X, y, batch_size=7, epochs=1000)
history_model
```

Train the model using the prepared data (X and y). Adjust the batch size and number of epochs as needed.

## 7. Evaluate Model Performance

```python
import matplotlib.pyplot as plt

history = history_model

# Retrieve loss and accuracy values from each epoch
loss = history.history['loss']
accuracy = history.history['accuracy']

# Set the x-axis as the number of epochs
epochs = range(1, len(loss
