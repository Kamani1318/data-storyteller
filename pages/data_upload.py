import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
from streamlit import components
import sys
# Create a custom SessionState class
class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Initialize SessionState
state = SessionState(data=None)

import json
import csv

def json_to_csv(json_data, csv_file):
    # Load JSON data
    with open(json_data, "r") as json_file:
        data = json.load(json_file)

    # Extract the keys from the JSON data
    keys = list(data[0].keys())

    # Write the JSON data to CSV
    with open(csv_file, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
        
def txt_to_csv(txt_file, csv_file):
    # Read TXT data
    with open(txt_file, "r") as txt_file:
        data = txt_file.readlines()

    # Write TXT data to CSV
    with open(csv_file, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["text"])  # Write header
        writer.writerows(zip(data))



def save_dataset_info(data, file_path):
    # Redirect the info output to the file
    with open(file_path, 'w') as file:
        sys.stdout = file  # Redirect stdout to the file
        data.info()

    # Reset stdout to the console
    sys.stdout = sys.__stdout__

    # Inform the user that the file was saved
    print(f"Dataset information saved to {file_path}")

def app():
    st.markdown("## Data Upload")

    # Upload the dataset and save as csv
    st.markdown("### Upload a csv file for analysis.")
    st.write("\n")

    # Code to read a single file
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'json', 'txt'])

    if uploaded_file is None:
        st.error("Upload a file")
        return

    if uploaded_file is not None:
        try:
            # Check file type
            file_type = uploaded_file.type
            # CSV file
            if file_type == "text/csv":
                data = pd.read_csv(uploaded_file)
            # XLSX file
            elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                data = pd.read_excel(uploaded_file)
            # JSON file
            elif file_type == "application/json":
                data = pd.read_json(uploaded_file)
            # TXT file
            elif file_type == "text/plain":
                data = pd.read_csv(uploaded_file, delimiter='\t')  # Assuming tab-separated TXT
            # Convert TXT to CSV
            if file_type == "text/plain":
                csv_file = uploaded_file.name.replace(".txt", ".csv")
                data.to_csv(csv_file, index=False)
                st.success("File converted to CSV: " + csv_file)
            # Convert JSON to CSV
            if file_type == "application/json":
                csv_file = uploaded_file.name.replace(".json", ".csv")
                data.to_csv(csv_file, index=False)
                st.success("File converted to CSV: " + csv_file)
            # Store data in state for further processing
            state.data = data

        except Exception as e:
            st.error("Error: " + str(e))

    save_dataset_info(state.data, "/Users/aryankamani/Projects/data-storyteller/data/metadata/main_data_info.txt")
    
    if state.data is not None:
        if st.button("Display Data"):
            st.dataframe(state.data)
            state.data.to_csv('data/main_data.csv', index=False)

        if st.button("Numerical and Categorical Columns"):
            numeric_cols = state.data.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = list(set(state.data.columns) - set(numeric_cols))
            st.markdown("<h3>Numerical Columns</h3>", unsafe_allow_html=True)
            for i in numeric_cols:
                st.markdown(f"- {i}")
            st.markdown("<h3>Categorical Columns</h3>", unsafe_allow_html=True)
            for i in categorical_cols:
                st.markdown(f"- {i}")
            numeric_cols_df = pd.DataFrame(numeric_cols, columns=["Numerical Columns"])
            numeric_cols_df.to_csv("data/metadata/numeric_columns.csv", index=False)
            categorical_cols_df = pd.DataFrame(categorical_cols, columns=["Categorical Columns"])
            categorical_cols_df.to_csv("data/metadata/categorical_columns.csv", index=False)
            st.markdown("""The above are the automated column types detected by the application in the data.
                            In case you wish to change the column types, head over to the **Column Change** section.""")