import joblib
from keras.models import load_model
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import os

# Create FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load the model and pipeline
model = load_model(r"BookingStatusModel.keras")
pipeline = joblib.load(r"preprocessor.joblib")

@app.get("/",response_class=HTMLResponse)
async def home(request:Request):
    return templates.TemplateResponse(request,"index2.html",{"request": request})

@app.post("/predict")
async def predict(request:Request,
    no_of_adults: int = Form(...),
    no_of_children: int = Form(...),
    no_of_weekend_nights: int = Form(...),
    no_of_week_nights: int = Form(...),
    type_of_meal_plan: int = Form(...),
    required_car_parking_space: int = Form(...),
    room_type_reserved: int = Form(...),
    lead_time: int = Form(...),
    arrival_year: int = Form(...),
    arrival_month: int = Form(...),
    arrival_date: int = Form(...),
    market_segment_type: int = Form(...),
    repeated_guest:int=Form(...),
    no_of_previous_cancellations: int = Form(...),
    no_of_previous_bookings_not_canceled: int = Form(...),
    avg_price_per_room: float=Form(...),
    no_of_special_requests:int=Form(...)):

    x = [no_of_adults, no_of_children, no_of_weekend_nights,
        no_of_week_nights, type_of_meal_plan, required_car_parking_space,
        room_type_reserved, lead_time, arrival_year, arrival_month,
        arrival_date, market_segment_type, repeated_guest,
        no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
       avg_price_per_room, no_of_special_requests]
    
    # clean and scale the data
    xpre = pipeline.transform([x])
    # predict the output
    res = model.predict(xpre) # predictions of neural network models are in probabilities

    # we finalise the prediction classes on basis of thresholds applied on probabilities
    if res[0][0]>0.5:
        res_op ='Approved'
    else:
        res_op = 'Not Approved'
    
    # Return results
    return templates.TemplateResponse(
    request,
    "index2.html",
    {
        "request": request,
        "prediction_text": res_op
    }
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app3:app", host="0.0.0.0", port=port)

# to run the app on local host: use below python command
# python app3.py