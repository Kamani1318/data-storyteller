# Load important libraries 
import pandas as pd
import streamlit as st 
import os
import numpy as np
import re
import sys
from dateutil.parser import parse

def save_dataset_info(data, file_path):
    # Redirect the info output to the file
    with open(file_path, 'w') as file:
        sys.stdout = file  # Redirect stdout to the file
        data.info()

    # Reset stdout to the console
    sys.stdout = sys.__stdout__

    # Inform the user that the file was saved
    print(f"Dataset information saved to {file_path}")
class Data_Information:

    def __init__(self,data):
      self.df = data
      self.column_types = data.dtypes
    def numerical_data_information(self):
        numerical_columns = self.column_types[self.column_types != object]
        data = []
        for column in self.df[numerical_columns.index.tolist()]:
            missing_count = self.df[column].isnull().sum()
            missing_percentage = (missing_count / len(self.df[column])) * 100
            distinct_count = self.df[column].nunique()
            distinct_percentage = (distinct_count / len(self.df[column])) * 100
            zeros_count = (self.df[column] == 0).sum()
            zeros_percentage = (zeros_count / len(self.df[column])) * 100
            negatives = (self.df[column] < 0).sum()
            negatives_percentage = (negatives / len(self.df[column])) * 100
            memory_size = self.df[column].memory_usage(index=False)

            data.append(
                {
                    "Column": column,
                    "Missing Count": missing_count,
                    "Missing Percentage": f"{missing_percentage:.2f}%",
                    "Distinct Count": distinct_count,
                    "Distinct Percentage": f"{distinct_percentage:.2f}%",
                    "Zeros Count": zeros_count,
                    "Zeros Percentage": f"{zeros_percentage:.2f}%",
                    "Negatives": negatives,
                    "Negatives Percentage": f"{negatives_percentage:.2f}%",
                    "Memory Size (kb)": memory_size,
                }
            )

        df = pd.DataFrame(data)

        st.markdown("## Numerical Information")
        st.table(df)

    def categorical_data_information(self):
        categorical_columns = self.column_types[self.column_types == object]
        data = []
        columns = ['Column', 'Frequency Count Percentage', 'Missing Value Count', 'Missing Percentage']

        for column in self.df[categorical_columns.index.tolist()]:
            frequency_counts = self.df[column].value_counts()
            value_counts_percentages = self.df[column].value_counts(normalize=True) * 100
            missing_count = self.df[column].isnull().sum()
            missing_percentage = (missing_count / len(self.df[column])) * 100

            data.append([column, value_counts_percentages, missing_count, missing_percentage])

        st.markdown("## Categorical Information")
        st.table(pd.DataFrame(data, columns=columns))

class NumericExtractor:
    def __init__(self, dataset, data_column):
        self.dataset = dataset
        self.column = data_column

    def extract_numeric_values(self):
        self.dataset[self.column] = self.dataset[self.column].apply(self.extract_numeric_value)
        return self.dataset

    @staticmethod
    def extract_numeric_value(value):
        numeric_value = re.findall(r'\d+\.\d+|\d+', str(value))
        if numeric_value:
            return float(numeric_value[0])
        else:
            return None
        
class DateColumnIdentifier:
  def __init__(self,data):
    self.data = data
  def identify_date_column(self):
    date_columns = []
    for column in self.data.columns:
      try:
        parse(self.data[column][0])
        date_columns.append(column)
        return date_columns  # Return the column name if successful
      except (ValueError, TypeError):
        pass
    return None

class DateConverter:
    def __init__(self, date_columns):
        self.date_columns = date_columns
        self.year_list = []
        self.month_list = []
        self.day_list = []

    def convert_date_column(self, data):
      if self.date_columns == None:
        return data
      for column in self.date_columns:
          for date_str in data[column]:
              date_obj = parse(date_str)
              self.year_list.append(date_obj.year)
              self.month_list.append(date_obj.month)
              self.day_list.append(date_obj.day)

      return self.year_list, self.month_list, self.day_list

    def add_date_columns(self, data):
      print(self.date_columns)
      if self.date_columns == None:
        return data
      year, month, day = self.convert_date_column(data)

        # Add new columns to the dataset
      data['Year'] = year
      data['Month'] = month
      data['Day'] = day

      for column in self.date_columns:
          data[column] = pd.to_datetime(data[column]).dt.strftime('%Y-%m-%d')

      return data
  
def compare_lists(list1, default_list):
  incorrect_list = []
  for value in list1:
    if value not in default_list:
      incorrect_list.append(value)
  return incorrect_list

def app():
    """This application is created to help the user change the metadata for the uploaded file. 
    They can perform merges. Change column names and so on.  
    """

    # Load the uploaded data 
    global copy_data
    copy_data = pd.read_csv('data/main_data.csv')
    copy_data.to_csv('data/duplicate_main_data.csv')
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        global data
        data = pd.read_csv('data/main_data.csv')
        st.dataframe(data)

        # Read the column meta data for this dataset 
        categorical_col = pd.read_csv('data/metadata/categorical_columns.csv')
        numeric_col = pd.read_csv('data/metadata/numeric_columns.csv')

        ''' Change the information about column types
            Here the info of the column types can be changed using dropdowns.
            The page is divided into two columns using beta columns 
        '''
        st.markdown("#### Change the information about column types")
        
        # Use two column technique 
        col1, col2 = st.columns(2)
        global name, type, current_type
        # Design column 1 
        current_type = 'numeric'
        name = col1.selectbox("Select Numeric Column", data.columns)
        for i in numeric_col[numeric_col.columns[0]]:
            if i == name:
                current_type = 'numeric'
        for i in categorical_col[categorical_col.columns[0]]:
            if i == name:
                current_type = 'categorical'
        column_options = ['numeric', 'categorical', 'datetime']
        
        current_index = column_options.index(current_type)
        type = col2.selectbox("Select Column Type", options=column_options, index = current_index)
        
        st.write("""Select your column name and the new type from the data.
                    To submit all the changes, click on *Submit changes* """)

        button_col1,button_col2 = st.columns(2)
        if button_col1.button("Change Column Type"): 
            if current_type == 'categorical' and type == 'numeric':
                try:
                    extractor = NumericExtractor(data, name)
                    numeric_dataset = extractor.extract_numeric_values()
                    data = numeric_dataset
                    data.to_csv('data/main_data.csv', index=False)
                except:
                    pass
                #data[name] = pd.to_numeric(data[name])
                #st.write(f"The data type of {name} is now {type}")
        try:
            dci = DateColumnIdentifier(data)
            date_columns = dci.identify_date_column()
            print('Date columns are: ', date_columns)
            converter = DateConverter(date_columns)
            converted_data = converter.add_date_columns(data)
            data = converted_data
            data.to_csv('data/main_data.csv', index=False)
        except:
            pass
        if button_col2.button("Provide Data Information"):
            try:
                data_info = Data_Information(data)
                data_info.numerical_data_information()
                data_info.categorical_data_information()
            except:
                pass
        st.dataframe(data)
                
        st.markdown(
            """
            <style>
            .stButton button {
                font-family: "Arial", sans-serif;
                font-size: 14px;
                font-weight: bold;
                color: #ffffff;
                background-color: #336699;
                padding: 0.5em 1em;
                border: none;
                border-radius: 4px;
            }
            .stButton button:hover {
                background-color: #002466;
                cursor: pointer;
            }
            </style>
            """,
            unsafe_allow_html=True
        ) 
                
        original_data = data.copy()
        drop_button = st.button("Drop Columns")
        original_data.to_csv('data/duplicate_main_data.csv', index=False)
        column_list_1 = st.multiselect("Select columns", options=data.columns, default=[], key="column_selection")

        if drop_button:
            data_filtered = original_data.drop(columns=column_list_1)
            st.dataframe(data_filtered)
            data_filtered.to_csv('data/main_data.csv', index=False)
            
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = list(set(data.columns) - set(numeric_cols))
        numeric_cols_df = pd.DataFrame(numeric_cols, columns=["Numerical Columns"])
        numeric_cols_df.to_csv("data/metadata/numeric_columns.csv", index=False)
        categorical_cols_df = pd.DataFrame(categorical_cols, columns=["Categorical Columns"])
        categorical_cols_df.to_csv("data/metadata/categorical_columns.csv", index=False)
        save_dataset_info(data, "/Users/aryankamani/Projects/data-storyteller/data/metadata/main_data_info.txt")