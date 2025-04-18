
import joblib
import pandas as pd
import numpy as np
import streamlit as st

Model = joblib.load('price_prediction.pkl')
inputs = joblib.load('inputs.pkl')

def prediction(Airline, Source, Destination, Duration, Total_Stops, Additional_Info, Day, Month, Dep_Hour, Dep_Min, Arrival_Hour, Arrival_Min):
    df = pd.DataFrame(columns=inputs)
    df.at[0, 'Airline'] = Airline
    df.at[0, 'Source'] = Source
    df.at[0, 'Destination'] = Destination
    df.at[0, 'Duration'] = Duration
    df.at[0, 'Total_Stops'] = Total_Stops
    df.at[0, 'Additional_Info'] = Additional_Info
    df.at[0, 'Day'] = Day
    df.at[0, 'Month'] = Month
    df.at[0, 'Dep_Hour'] = Dep_Hour
    df.at[0, 'Dep_Min'] = Dep_Min
    df.at[0, 'Arrival_Hour'] = Arrival_Hour
    df.at[0, 'Arrival_Min'] = Arrival_Min
    result = Model.predict(df)[0]
    return result

def Main():
    st.title('Flight Price Prediction')

    Airline = st.selectbox('Airline', ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia', 'Vistara Premium economy', 'Multiple carriers Premium economy', 'Trujet'])
    Source = st.selectbox('Source', ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'])
    Destination = st.selectbox('Destination', ['New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad'])
    Duration = st.slider('Duration', min_value= 60.0 , max_value=3000.0, step=1.0, value = 60.0)
    Total_Stops = st.slider('Total_Stops', min_value= 0.0, max_value= 4.0, step= 1.0, value= 1.0)
    Additional_Info = st.selectbox('Additional_Info', ['No info', 'In-flight meal not included', 'No check-in baggage included', '1 Long layover', 'Change airports', 'Red-eye flight'])
    Day = st.slider('Day', min_value= 1.0 , max_value= 31.0 , step= 1.0, value = 13.0)
    Month = st.slider('Month', min_value= 1.0, max_value= 12.0, step= 1.0, value = 10.0)
    Dep_Hour = st.slider('Dep_Hour', min_value= 0.0, max_value= 23.0 , step= 1.0, value = 10.0)
    Dep_Min = st.slider('Dep_Min', min_value= 0.0, max_value= 59.0, step= 1.0, value= 10.0)
    Arrival_Hour = st.slider('Arrival_Hour', min_value= 0.0, max_value= 23.0, step= 1.0, value= 10.0)
    Arrival_Min = st.slider('Arrival_Min', min_value= 0.0, max_value= 59.0, step= 1.0, value= 10.0)


    if st.button('Predict'):
        result = prediction(Airline, Source, Destination, Duration, Total_Stops, Additional_Info, Day, Month, Dep_Hour, Dep_Min, Arrival_Hour, Arrival_Min)
        st.text(f"Your Flight Price is {result}")

Main()
