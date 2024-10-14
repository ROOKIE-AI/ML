import os
import streamlit as st
import pandas as pd
 
st.write("""
# My first app
Hello *world!*
""")

root = os.path.abspath(os.path.dirname(__file__))

df = pd.read_csv(os.path.join(root, "train.csv"))
st.slider('button', disabled=False, label_visibility="visible")