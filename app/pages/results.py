import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_icon = '🦋')

dumps = r'/home/sauravs/heroku/sine-curve-ml/dumps/'

hist = pd.read_csv(dumps + 'train_history.csv')  # Adding values to the card
epoch = [i for i in range(len(hist))]
hist['epoch'] = epoch

st.write("Loss Graph")
fig = plt.figure(figsize=(20,10))
plt.plot(hist['epoch'], hist['mse'], label='Train')
plt.plot(hist['epoch'], hist['val_mse'], label='Validation')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.title('Loss Graph')
plt.legend(loc='best')
st.write(fig)