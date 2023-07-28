import os
import streamlit as st
import numpy as np
from PIL import  Image

# Custom imports 
from multipage import MultiPage
from pages import data_upload, machine_learning, metadata, data_visualize, redundant, Final_Preprocessed_Data # import your pages here

# Create an instance of the app 
app = MultiPage()

# Title of the main page
display = Image.open('/Users/aryankamani/Data-Analysis-Tool/Machine Learning in Healthcare.png')
display = np.array(display)
col1, col2 = st.columns(2)
col1.image(display, width = 250)
col2.title("Data Preprocessing Application")

# Add all your application here
app.add_page("Upload Data", data_upload.app)
app.add_page("Change Metadata", metadata.app)
app.add_page("Final Preprocessed Data",Final_Preprocessed_Data.app )
app.add_page("Data Visualisation",data_visualize.app)
app.add_page("Machine Learning", machine_learning.app)
app.add_page("Y-Parameter Optimization",redundant.app)


# The main app
app.run()

