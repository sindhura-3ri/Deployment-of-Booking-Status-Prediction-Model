import streamlit as st
import joblib
import keras
import pandas as pd
from keras.models import load_model

# Provide Tab title
st.set_page_config("Deployment Project")
# Provide the page title
st.title("Booking Status Prediction Model")
# Mention description of the problem statement or if you want to mention your name
st.subheader("This project takes multiple details as input and predicts whether booking status is approved or not")
st.subheader("By Sindhura Kuntamukkula")

# Consider below features as input fields in the webpage.
# 'no_of_adults', 'no_of_children', 'no_of_weekend_nights',
#        'no_of_week_nights', 'type_of_meal_plan', 'required_car_parking_space',
#        'room_type_reserved', 'lead_time', 'arrival_year', 'arrival_month',
#        'arrival_date', 'market_segment_type', 'repeated_guest',
#        'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
#        'avg_price_per_room', 'no_of_special_requests'

adult_count = st.number_input(label="Number of Adults",min_value=1,max_value=15,step=1)
children_count = st.number_input(label="Number of children",min_value=1,max_value=15,step=1)
weekend_nights = st.number_input(label="Number of weekend nights",min_value=1,max_value=15,step=1)
week_nights = st.number_input(label="Number of Weekday nights",min_value=1,max_value=15,step=1)
type_of_meal_plan = st.number_input(label="Type of meal plan")
car_parking_required = st.number_input(label="required_car_parking_space")
room_type_reserved = st.number_input(label="Room type")
lead_time= st.number_input(label="Lead time")
arrival_year = st.number_input(label="Arrival Year")
arrival_month = st.number_input(label="Arrival Month")
arrival_date = st.number_input(label="Arrival Date")
market_segment_type = st.number_input(label="Market Segment Type")
repeated_guest = st.number_input(label="Repeated Guest")
no_of_previous_cancellations = st.number_input(label="Total Previous cancellations")
no_of_previous_bookings_not_canceled = st.number_input(label="Total Previous non cancellations")
avg_price_per_room = st.number_input(label="Average price per room")
no_of_special_requests= st.number_input("Total special requests")

# Create a button with name Predict. 
submit = st.button("Predict the Booking status here")

# Load the necessary model and pipeline
# syntax: file_name = load_model(r"pass the path of model")
# file_name = joblib.load(r"pass the path of pipeline")
model = load_model(r"/workspaces/Deployment-of-Booking-Status-Prediction-Model/BookingStatusModel.keras")
pipeline = joblib.load(r"/workspaces/Deployment-of-Booking-Status-Prediction-Model/preprocessor.joblib")

# the logic to be executed when predict button is clicked(submit becomes True)
if submit: # if submit==True
    dct = {
        'no_of_adults':[adult_count],
        'no_of_children':[children_count],
        'no_of_weekend_nights': [weekend_nights],
       'no_of_week_nights': [week_nights],
       'type_of_meal_plan':[type_of_meal_plan], 
       'required_car_parking_space':[car_parking_required],
       'room_type_reserved':[room_type_reserved], 
       'lead_time':[lead_time], 
       'arrival_year' : [arrival_year], 
       'arrival_month' : [arrival_month],
       'arrival_date' : [arrival_date], 
       'market_segment_type' : [market_segment_type],
        'repeated_guest' : [repeated_guest],
       'no_of_previous_cancellations' : [no_of_previous_cancellations], 
       'no_of_previous_bookings_not_canceled' : [no_of_previous_bookings_not_canceled],
       'avg_price_per_room' : [avg_price_per_room], 
       'no_of_special_requests' : [no_of_special_requests]
    }
    xnew = pd.DataFrame(dct) # all the inputs provided by user is gathered in a dictionary. We are converting dict data into dataframe
    xnew_pre = pipeline.transform(xnew) # data cleaning + scaling of numerical data
    preds = model.predict(xnew_pre)
    # preds: output: [[0]] : preds[0][0]
    # preds : output : [0] : preds[0]

    if preds[0]>0.5:
        res_op = "Booking will be Approved"
    else:
        res_op = "Booking will NOT be Approved"
    
    st.subheader(f"Predictions are: {res_op}")



