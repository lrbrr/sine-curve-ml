import math
import config
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

x = []
sin_x = []

for i in range(config.train_len):
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