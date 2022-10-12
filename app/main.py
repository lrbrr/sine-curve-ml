import time
import math
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

from css.cssHelper import local_css, remote_css, icon

data = st.container()

# data, apply, play = st.tabs(["🗃 Data"])

local_css("./app/css/style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

with data:
    st.title('Predict Sine values using NN')
    
    train_len = st.slider('No. of train samples: 🗃 Data', 10, 500)

    x = []
    sin_x = []

    for i in range(train_len):
        dataPoints = abs((np.random.randint(0, 100) / 10) - 6.2832)
        if dataPoints not in x:
            x.append(dataPoints)                        # Inputs
            sin_x.append(math.sin(dataPoints))          # Outputs

    x = np.array(x)
    sin_x = np.array(sin_x)

    # To know why this is important -> https://youtu.be/V2QlTmh6P2Y

    x = x.reshape((x.shape[0],1))
    sin_x = sin_x.reshape((sin_x.shape[0],1))

    fig = plt.figure(figsize=(20,10))
    plt.scatter(x, sin_x, c='r')       # c=’r’ because default(blue) is boring
    plt.xlabel('x')
    plt.ylabel('Sin(x)')
    plt.title('Sexy Sine')

    st.write(fig)

    train_click = st.button('Train using this sample data')
    
    if train_click:
        st.spinner(text='Training in Progress')
        time.sleep(5)
        st.snow()
        st.success("Scroll down")

