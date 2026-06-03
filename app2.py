from flask import Flask,render_template,request
import joblib
from keras.models import load_model

# create flask app
flask_app = Flask(__name__)

# Load the model and pipeline
model = load_model(r"BookingStatusModel.keras")
pipeline = joblib.load(r"preprocessor.joblib")

# API endpoints: GET : bring the information, POST: get and push it to the server

@flask_app.route("/")
def Home(): # this is the default home page
    return render_template("index.html")

@flask_app.route("/predict",methods="POST") # WHAT SHOULD HAPPEN WHEN USER CLICKS ON PREDICT
def predict():
    # convert numerical features into float to fetch the data as forms consider data in string format
    no_of_adults = float(request.form["Number of Adults"])
    Number_of_Children = float(request.form["Number of Children"])
    Number_of_weekend_nights = float(request.form["Number of weekend nights"])
    Number_of_Weekday_nights = float(request.form["Number of Weekday nights"])
    Type_of_meal_plan = float(request.form["Type of meal plan"])
    required_car_parking_space = float(request.form["required_car_parking_space"])
    Room_type= float(request.form["Room type"])
    Lead_Time=float(request.form["Lead Time"])
    Arrival_Year = float(request.form["Arrival_Year"])
    Arrival_Month = float(request.form["Arrival_Month"])
    Arrival_Day = float(request.form["Arrival_Day"])
    MarketSegmentType = float(request.form["Market Segment Type"])
    RepeatedGuest = float(request.form["Repeated Guest"])
    Total_Previous_cancellations = float(request.form["Total Previous cancellations"])
    Total_Previous_noncancellations = float(request.form["Total Previous non cancellations"])
    Totalspecialrequests = float(request.form["Total special requests"])
    Averagepriceperroom = float(request.form["Average price per room"])

    # load this form gathered data into a dictionary format
    dct = {
        'no_of_adults':[no_of_adults],
        'no_of_children':[Number_of_Children],
        'no_of_weekend_nights': [Number_of_weekend_nights],
       'no_of_week_nights': [Number_of_Weekday_nights],
       'type_of_meal_plan':[Type_of_meal_plan], 
       'required_car_parking_space':[required_car_parking_space],
       'room_type_reserved':[Room_type], 
       'lead_time':[Lead_Time], 
       'arrival_year' : [Arrival_Year], 
       'arrival_month' : [Arrival_Month],
       'arrival_date' : [Arrival_Day], 
       'market_segment_type' : [MarketSegmentType],
        'repeated_guest' : [RepeatedGuest],
       'no_of_previous_cancellations' : [Total_Previous_cancellations], 
       'no_of_previous_bookings_not_canceled' : [Total_Previous_noncancellations],
       'avg_price_per_room' : [Averagepriceperroom], 
       'no_of_special_requests' : [Totalspecialrequests]
    }
    # convert dictionary into dataframe
    xnew = pd.DataFrame(dct)
    # apply pipeline and transform xnew
    xnew_pre = pipeline.transform(xnew)
    # model predictions
    preds = model.predict(xnew_pre)
    if preds[0]>0.5:
        res_op = "Booking Approved"
    else:
        res_op = "Booking Not Approved"
    
    return render_template("index.html",prediction_text=res_op)

if __name__=="__main__":
    flask_app.run(debug=True)




