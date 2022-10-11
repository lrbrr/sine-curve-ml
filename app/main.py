import config
import streamlit as st

from css.cssHelper import local_css, remote_css, icon
from sineTraining import fig

header = st.container()
form = st.container()
result = st.container()
plots = st.container()

local_css("./app/css/style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

with header:
    st.title('Predict Sine values using NN')

with form:
    st.text('Generate random sine values for training')
    config.train_len = st.number_input('No. of train samples', 100)
    st.write(fig)

with result:
    st.button('Train using this sample data')