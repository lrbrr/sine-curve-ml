import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from css.cssHelper import local_css, remote_css, icon

st.set_page_config(page_title = 'Multipage App', page_icon = '🦋')

local_css("./app/css/style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

st.title('Predict Sine values using NN')

train_len = st.slider('🗃 Data No. of train samples:', 10, 500)

x = []
sin_x = []

for i in range(train_len):
    dataPoints = abs((np.random.randint(0, 100) / 10) - 6.2832)
    if dataPoints not in x:
        x.append(dataPoints)                        # Inputs
        sin_x.append(math.sin(dataPoints))          # Outputs

# store data into dataset
dumps = r'/home/sauravs/heroku/sine-curve-ml/dumps/'

df = pd.DataFrame({'x': x, 'sin(x)': sin_x})
df.to_csv(dumps + "dataset.csv", index=False)

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
st.write(f"Out of {train_len} data points, only {len(x)} data points are usable.")