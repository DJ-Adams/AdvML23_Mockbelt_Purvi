# model deployment using streamlit

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, sys
from sklearn import set_config
set_config(transform_output='pandas')

# Load the filepaths
FILEPATHS_FILE = 'config/filepaths.json'
with open(FILEPATHS_FILE) as f:
    FPATHS = json.load(f)
    
# Define the load train or test data function with caching
@st.cache_data
def load_Xy_data(fpath):
    return joblib.load(fpath)
    
@st.cache_resource
def load_model_ml(fpath):
    return joblib.load(fpath)
    
### Start of App
st.title('House Prices Prediction')
# # Include the banner image
# st.image(FPATHS['images']['banner'])


# Load training data
X_train, y_train = load_Xy_data(fpath=FPATHS['data']['ml']['train'])
# Load testing data
X_test, y_test = load_Xy_data(fpath=FPATHS['data']['ml']['test'])
# Load model
linreg = load_model_ml(fpath = FPATHS['models']['linear_regression'])

# Add text for entering features
st.subheader("Select values using the sidebar on the left.\n Then check the box below to predict the price.")
st.sidebar.subheader("Enter House Features For Prediction")

# # Create widgets for each feature
#bedrooms
Bedrooms = st.sidebar.slider('Bedrooms',
                            min_value = X_train['bedrooms'].min(),
                            max_value = X_train['bedrooms'].max(),
                            step = 1, value = 3)

#bathrooms
Bathrooms = st.sidebar.slider('Bathrooms',
                             min_value = X_train['bathrooms'].min(),
                             max_value = X_train['bathrooms'].max(),
                             step = .25, value = 2.5)

#sqft_living
sqft_living = st.sidebar.number_input('Sqft Living Area',
                                     min_value=100,
                                     max_value=X_train['sqft_living'].max(),
                                     step=150, value=2500)


# Define function to convert widget values to dataframe
def get_X_to_predict():
    X_to_predict = pd.DataFrame({'Bedroom': Bedrooms,
                                 'Bathroom':Bathrooms,
                                 'Living Area Sqft':sqft_living},
                             index=['House'])
    return X_to_predict
    
def get_prediction(model,X_to_predict):
    return  model.predict(X_to_predict)
    
if st.checkbox("Predict"):
    
    X_to_pred = get_X_to_predict()
    new_pred = get_prediction(linreg, X_to_pred)
    
    st.markdown(f"> #### Model Predicted Price = ${new_pred:,.0f}")
    
else:
    st.empty()

