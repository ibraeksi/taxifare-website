import streamlit as st

'''
# TaxiFareModel front
'''

st.markdown('''
Remember that there are several ways to output content into your web page...

Either as with the title by just creating a string (or an f-string). Or as with this paragraph using the `st.` functions
''')

'''
## Here we would like to add some controllers in order to ask the user to select the parameters of the ride

1. Let's ask for:
- date and time
- pickup longitude
- pickup latitude
- dropoff longitude
- dropoff latitude
- passenger count
'''

'''
## Once we have these, let's call our API in order to retrieve a prediction

See ? No need to load a `model.joblib` file in this app, we do not even need to know anything about Data Science in order to retrieve a prediction...

🤔 How could we call our API ? Off course... The `requests` package 💡
'''

url = 'https://taxifare-688954311280.europe-west10.run.app/predict'
# url = 'https://taxifare.lewagon.ai/predict'

if url == 'https://taxifare.lewagon.ai/predict':

    st.markdown('Maybe you want to use your own API for the prediction, not the one provided by Le Wagon...')

'''

2. Let's build a dictionary containing the parameters for our API...

3. Let's call our API using the `requests` package...

4. Let's retrieve the prediction from the **JSON** returned by the API...

## Finally, we can display the prediction to the user
'''

params = {"pickup_datetime":"2014-07-06 19:18:00", "pickup_longitude":-73.950655,
          "pickup_latitude":40.783282, "dropoff_longitude":-73.984365,
          "dropoff_latitude":40.769802, "passenger_count":2}

import requests
import numpy as np
import urllib
import matplotlib.pyplot as plt
import PIL
from pathlib import Path

response = requests.get(url, params=params).json()

st.write("The predicted fare price: ", np.round(response['fare'],2), "$")


# Let's check NYC bouding boxes
# Load image of NYC map
bounding_boxes = (-74.3, -73.7, 40.5, 40.9)

url = 'https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/nyc_-74.3_-73.7_40.5_40.9.png'
nyc_map = np.array(PIL.Image.open(urllib.request.urlopen(url)))

def plot_on_map(params, BB, nyc_map, s=10, alpha=0.2):
    fig, ax = plt.subplots(1, 1, figsize=(16,10))

    ax.scatter(params["pickup_longitude"], params["pickup_latitude"], zorder=1, alpha=alpha, c='red', s=s)
    ax.scatter(params["dropoff_longitude"], params["dropoff_latitude"], zorder=1, alpha=alpha, c='blue', s=s)
    ax.set_xlim((BB[0], BB[1]))
    ax.set_ylim((BB[2], BB[3]))
    ax.set_title('Pickup & Dropoff Locations')
    ax.imshow(nyc_map, zorder=0, extent=BB)

    return fig

# Plot training data on map
fig = plot_on_map(params, bounding_boxes, nyc_map, s=15, alpha=1)

st.pyplot(fig)
