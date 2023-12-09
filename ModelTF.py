import streamlit as st
import time
from st_pages import Page, show_pages, add_page_title
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from keras.models import Sequential, save_model

df=pd.read_csv('diabetes.csv')
df['Glucose']=df['Glucose'].replace(0, df['Glucose'].mean())
df['BloodPressure']=df['BloodPressure'].replace(0, df['BloodPressure'].mean())
df['SkinThickness']=df['SkinThickness'].replace(0, df['SkinThickness'].mean())
df['Insulin']=df['Insulin'].replace(0, df['Insulin'].mean())
df['BMI']=df['BMI'].replace(0, df['BMI'].mean())
df['Age']=df['Age'].replace(0, df['Age'].mean())


X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values
X = df.drop(['BloodPressure','SkinThickness','Insulin','DiabetesPedigreeFunction'], axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(
X, y, 
test_size=0.2, random_state=42
)
X_train,y_train
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled
tf.random.set_seed(42)
model = tf.keras.Sequential([
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(256, activation='relu'),
tf.keras.layers.Dense(256, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(
loss=tf.keras.losses.binary_crossentropy,
optimizer=tf.keras.optimizers.Adam(lr=0.03),
metrics=[
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]
)
history = model.fit(X_train_scaled, y_train, epochs=100)

save_model(model, 'flower_model_trained.hdf5')
print("Model Saved")
