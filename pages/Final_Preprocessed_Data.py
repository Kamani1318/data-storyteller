import streamlit as st
import pandas as pd
import numpy as np
import os
import category_encoders as ce
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method='iqr', multiplier=1.5):
        self.method = method
        self.multiplier = multiplier
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.imputer.fit(X)
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X = self.imputer.transform(X)

        if self.method == 'iqr':
            q1 = np.percentile(X, 25, axis=0)
            q3 = np.percentile(X, 75, axis=0)
            iqr = q3 - q1
            lower_bound = q1 - self.multiplier * iqr
            upper_bound = q3 + self.multiplier * iqr
            X = np.where(X < lower_bound, lower_bound, X)
            X = np.where(X > upper_bound, upper_bound, X)
        elif self.method == 'std':
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            lower_bound = mean - self.multiplier * std
            upper_bound = mean + self.multiplier * std
            X = np.where(X < lower_bound, lower_bound, X)
            X = np.where(X > upper_bound, upper_bound, X)
        else:
            raise ValueError("Invalid method. Supported methods are 'iqr' and 'std'.")

        X = self.scaler.transform(X)
        return X
    
class HandleDuplicateValues(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.duplicated_columns = []

    def fit(self, X, y=None):
        duplicated_columns = []
        if isinstance(X, pd.DataFrame):
            duplicated_columns = X.columns[X.duplicated()]
        self.duplicated_columns = list(duplicated_columns)
        return self
    

class Normalization(BaseEstimator, TransformerMixin):
  def __init__(self):
    self.scaler = MinMaxScaler(feature_range= (0,1))
  def fit(self, X, y=None):
    self.scaler.fit(X)
    return self

  def transform(self, X):
    return self.scaler.transform(X)

    
class OneHotEncoderWithoutMissing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns = pd.get_dummies(X).columns
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X_encoded = pd.get_dummies(X)

        # Remove the 'missing' column if present
        if 'missing' in self.columns:
            X_encoded = X_encoded.drop('missing', axis=1)

        # Reindex columns to match the original encoding
        X_encoded = X_encoded.reindex(columns=self.columns, fill_value=0)
        # Rename columns
        X_encoded.columns = self.columns

        return X_encoded
    
def pipeline_processsing(data):
    column_types = data.dtypes
    categorical_columns = column_types[column_types == object]  # Adjust the data type if needed
    numerical_columns = column_types[column_types !=object]
    categorical_columns_list = categorical_columns.index.tolist()
    numerical_columns_list = numerical_columns.index.tolist()
    unique_categories = []
    for column in categorical_columns_list:
        unique_categories.extend(data[column].unique().tolist())
    si = SimpleImputer(missing_values = np.nan,strategy = 'mean')

    outlierhandler = OutlierHandler(method = 'iqr', multiplier = 1.5)
    
    numeric_processor = Pipeline(steps = [
        ("imputation mean",si),
        ("outlier handler", outlierhandler),
        #("duplicate values handler", HandleDuplicateValues()),
        ("scaler",StandardScaler()),
        ("normalisation",Normalization())
        ])
    categorical_processor = Pipeline(steps = [
        ("imputation constant",SimpleImputer(fill_value = "missing",strategy = 'constant')),
        ("one hot encoder", OneHotEncoderWithoutMissing())
        ])

    preprocessor = ColumnTransformer(
        [("categorical",categorical_processor,categorical_columns_list),
        ("numerical", numeric_processor,numerical_columns_list)]
        )
    processed_data = preprocessor.fit_transform(data)
    processed_data = pd.DataFrame(processed_data)
    column_list = unique_categories + numerical_columns_list
    processed_data.columns = column_list
    return processed_data

def download():
    st.title("Dataset Downloader")
    
    if st.button("Download Data"):
        # Provide the file path to be downloaded
        file_path = "data/main_data.csv"

        # Trigger the download
        st.download_button(label="Click to Download", data=file_path, file_name="data.csv")

def app():
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        global data
        data = pd.read_csv('data/main_data.csv')
        st.header("Putting the Data in the Pipeline for Preprocessing")
        st.dataframe(data)
        
        
        if st.button("Count Encoding"):
            var_list= st.multiselect("Select the variables for Count Encoding", options=data.columns)
            if var_list is None:
                st.write("Select a variable: ")
            st.write(var_list)
            count_data = data.copy()
            for var in var_list:
                count_data[var] = data.groupby(var)[var].transform('count')
            data = count_data
            st.write("Count encoding done")
            st.dataframe(data)
            data.to_csv('data/main_data.csv')
            
        
        
        if st.button("Process the Data"):
            with st.spinner("Preprocessing data..."):
                data = pd.read_csv('data/main_data.csv')
                pprocessed_data = pipeline_processsing(data)
                st.dataframe(pprocessed_data)
                pprocessed_data.to_csv('data/main_data.csv', index=False)
                download()