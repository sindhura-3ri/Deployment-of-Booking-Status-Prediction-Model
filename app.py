import streamlit as st
import joblib
import keras
import pandas as pd

# Provide Tab title
st.set_page_config("Deployment Project")
# Provide the page title
st.title("Booking Status Prediction Model")
# Mention description of the problem statement or if you want to mention your name
st.subheader("This project takes multiple details as input and predicts whether booking status is approved or not")
st.subheader("By Sindhura Kuntamukkula")



