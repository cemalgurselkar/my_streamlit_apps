import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.write("""
# Penguin Prediction App
""")

uploaded_file = st.sidebar.file_uploader("Upload your csv file", type=['csv'])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_parameters():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length = st.sidebar.slider('Bill Length (mm)',10.0,60.0,43.4)
        bill_depth = st.sidebar.slider('Bill Depth (mm)', 10.0,30.0,17.3)
        flipper_length = st.sidebar.slider('Flipper Length (mm)', 150.0, 250.0, 201.0)
        body_mass = st.sidebar.slider('Body Mass (g)', 2700.0, 6300.0, 3603.5)
        data = {'island':island,
                'bill_length_mm': bill_length,
                'bill_depth_mm':bill_depth,
                'flipper_length_mm':flipper_length,
                'body_mass_g':body_mass,
                'sex':sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_parameters()

penguin_raw = pd.read_csv('datas/penguin_data.csv')
penguins = penguin_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)

encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col],prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

df = df[:1]

st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

load_clf = pickle.load(open('model.pkl','rb'))
prediction = load_clf.predict(df)
prediction_prob = load_clf.predict_proba(df)

st.subheader('Prediction')
penguin_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguin_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_prob)