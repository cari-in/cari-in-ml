# Building a Machine Learning Model for Bali Tourism Prediction

In this guide, we will explain the step-by-step process of building a machine learning model that can be used to predict tourism in Bali. The provided source code utilizes the following libraries: Pandas, NumPy, scikit-learn, and TensorFlow.

## 1. Import Required Libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
```

The first step is to import the necessary libraries for constructing and training the machine learning model.

## 2. Read the Data

```python
df = pd.read_csv("/Copy of Data_Wisata_Bali - Sheet1 (2).csv")
```

Next, we read the Bali tourism data from a CSV file using the Pandas library. Ensure that the file path points to the correct location of the file you wish to read.

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

Use scikit-learn's LabelEncoder to convert categorical values into numerical values that can be processed by the machine learning model.

```python
df_le = pd.DataFrame()
df_le["Wilayah_Encoded"] = label_encoder.fit_transform(df["Wilayah"])
df_le["Panorama_Encoded"] = label_encoder.fit_transform(df["Panorama"])
df_le["Tipe Rekreasi_Encoded"] = label_encoder.fit_transform(df["Tipe Rekreasi"])
df_le["Target Prediksi_Encoded"] = label_encoder.fit_transform(df["Target Prediksi (y)"])
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
model.add(Dense(32, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(y.unique().shape[0], activation='softmax'))
```

Construct a neural network model using TensorFlow. The model consists of three layers: two hidden layers with ReLU activation functions and one output layer with a softmax activation function.

## 5. Compile the Model

```python
lr = 0.001  # Desired learning rate
optimizer = SGD(learning_rate=lr)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

Compile the model using the Stochastic Gradient Descent (SGD) optimizer and the appropriate loss function for multiclass classification.

## 6. Train the Model

```python
model.fit(X, y, batch_size=7, epochs=10)
```

Train the model using the previously split X and y data. Adjust the batch size and number of epochs according to your needs.

## 7. Making Predictions

```python
input = np.array([[4, 1, 10]])
input.shape
y_pred = model.predict(np.array([[0, 1, 2]]))
print(np.argmax(y_pred))
```

Make predictions using the trained model. In this example, we provide a sample input and predict the corresponding
