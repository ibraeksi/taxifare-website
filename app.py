import streamlit as st
import requests
import numpy as np
import urllib
import matplotlib.pyplot as plt
import PIL

'''
# NYC Taxi Fare Prediction
'''

with st.form("predict"):
    pickup_datetime = st.text_input("Enter Pickup Datetime","2014-07-06 19:18:00")
    pickup_longitude = st.number_input("Pickup Longitude", format="%.6f", value=-73.950655)
    pickup_latitude = st.number_input("Pickup Latitude", format="%.6f", value=40.783282)
    dropoff_longitude = st.number_input("Dropoff Longitude", format="%.6f", value=-73.984365)
    dropoff_latitude = st.number_input("Dropoff Latitude", format="%.6f", value=40.769802)
    passenger_count = st.number_input("Passenger Count", value=2)
    submit = st.form_submit_button("Predict 🚀")

url = 'https://taxifare-688954311280.europe-west10.run.app/predict'

if submit:
    params = {"pickup_datetime":pickup_datetime, "pickup_longitude":pickup_longitude,
            "pickup_latitude":pickup_latitude, "dropoff_longitude":dropoff_longitude,
            "dropoff_latitude":dropoff_latitude, "passenger_count":passenger_count}

    response = requests.get(url, params=params).json()
    fare = response['fare']

    st.success(f"The predicted fare price: **${fare:.2f}**")

    # Load image of NYC map
    bounding_boxes = (-74.3, -73.7, 40.5, 40.9)

    url = 'https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/nyc_-74.3_-73.7_40.5_40.9.png'
    nyc_map = np.array(PIL.Image.open(urllib.request.urlopen(url)))

    def plot_on_map(params, BB, nyc_map, s=10, alpha=0.2):
        fig, ax = plt.subplots(1, 1, figsize=(16,10))

        ax.scatter(params["pickup_longitude"], params["pickup_latitude"], zorder=1, alpha=alpha, c='red', s=s)
        ax.annotate("Pickup", (params["pickup_longitude"], params["pickup_latitude"]), fontsize=12, color='red')
        ax.scatter(params["dropoff_longitude"], params["dropoff_latitude"], zorder=1, alpha=alpha, c='blue', s=s)
        ax.annotate("Dropoff", (params["dropoff_longitude"], params["dropoff_latitude"]), fontsize=12, color='blue')
        ax.set_xlim((BB[0], BB[1]))
        ax.set_ylim((BB[2], BB[3]))
        ax.set_title('Pickup & Dropoff Locations')
        ax.imshow(nyc_map, zorder=0, extent=BB)

        return fig

    # Plot training data on map
    fig = plot_on_map(params, bounding_boxes, nyc_map, s=20, alpha=1)

    st.pyplot(fig)
