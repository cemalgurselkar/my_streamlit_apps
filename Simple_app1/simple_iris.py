import streamlit as st
import numpy as np
import pandas as pd 
from sklearn import datasets 
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction
""")

st.sidebar.header('Parameters')

def user_parameters():
    sepal_length = st.sidebar.slider('sepal_lenght', 1.0, 10.0, 4.0)
    sepal_witdh = st.sidebar.slider('sepal_witdh', 1.0, 10.0, 3.0)
    petal_length = st.sidebar.slider('petal_length', 1.0,10.0,2.0)
    petal_witdh = st.sidebar.slider('petal_witdh', 1.0,5.0,1.2)

    data = {'sepal_length': sepal_length,
            'sepa_width': sepal_witdh,
            'petal_length': petal_length,
            'petal_witdh': petal_witdh}

    features = pd.DataFrame(data, index=[0])
    return features

df = user_parameters()

st.subheader('User Iris parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(X,y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Predictions')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)