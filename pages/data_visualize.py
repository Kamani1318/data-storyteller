import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Visualisation:
    
    def __init__(self, data, column_list, chart_type='boxplot'):
        self.df = data
        self.column1 = column_list[0]
        self.column2 = column_list[1]
        self.chart_type = chart_type

    def visualise(self):
        st.header(self.chart_type)
        fig, ax = plt.subplots()
        numeric_data = self.df[(self.df.select_dtypes(include=np.number).columns.tolist())]
        corr_matrix = numeric_data.corr()
        st.markdown("<h3>Correlation Matrix</h3>", unsafe_allow_html=True)
        st.write(corr_matrix)
        if self.chart_type == 'boxplot':
            sns.boxplot(x=self.column1, y=self.column2, data=self.df)
            ax.set_title(f'{self.column1} vs {self.column2}')

        elif self.chart_type == 'scatter plot':
            sns.scatterplot(x=self.df[self.column1], y=self.df[self.column2])
            ax.set_title(f'{self.column1} vs {self.column2}')
            

        elif self.chart_type == 'histogram':
            sns.histplot(data=self.df, x=self.column1, hue=self.column2, multiple='stack')
            ax.set_title(f'{self.column1} Distribution by {self.column2}')
        

        elif self.chart_type == 'barplot':
            sns.barplot(x=self.column1, y=self.column2, data=self.df)
            ax.set_title(f'{self.column1} vs {self.column2}')
         

        elif self.chart_type == 'line plot':
            sns.lineplot(x=self.column1, y=self.column2, data=self.df)
            ax.set_title(f'{self.column1} vs {self.column2}')
        

        elif self.chart_type == 'violin plot':
            sns.violinplot(x=self.column1, y=self.column2, data=self.df)
            ax.set_title(f'{self.column1} vs {self.column2}')
         

        elif self.chart_type == 'heatmap':
            pivot_table = self.df.pivot_table(values=self.column2, index=self.column1)
            sns.heatmap(pivot_table, annot=True, cmap='YlGnBu')
            ax.set_title(f'{self.column1} vs {self.column2}')
        

        elif self.chart_type == 'area plot':
            sns.lineplot(x=self.column1, y=self.column2, data=self.df)
            plt.fill_between(self.df[self.column1], self.df[self.column2], alpha=0.5)
            ax.set_title(f'{self.column1} vs {self.column2}')
            

        elif self.chart_type == 'QQ plot':
            sns.qqplot(self.df[self.column1])
            ax.set_title(f'QQ Plot for {self.column1}')
        st.pyplot(fig)
        
        

def app():
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        # df_analysis = pd.read_csv('data/2015.csv')
        df_analysis = pd.read_csv('data/main_data.csv')
        # df_visual = pd.DataFrame(df_analysis)
        df_visual = df_analysis.copy()
        col1, col2 , col3 = st.columns(3)
        column_list = df_visual.columns.tolist()
        chart_list = ['boxplot','scatter plot','histogram','barplot','line plot','violin plot','heatmap','area plot','QQ plot']
        column1 = col1.selectbox("Select a column",column_list, key = "column1")
        column2 = col2.selectbox("Select a column", column_list, key = "column2")
        chart_type = col3.selectbox("Select a chart type", chart_list, key = "chart_type")
        list_of_columns = []
        list_of_columns.append(column1)
        list_of_columns.append(column2)
        if len(list_of_columns) < 2:
            st.error("Cannor visualise without two variables")
            return
        vis = Visualisation(df_visual,list_of_columns, chart_type = chart_type)
        vis.visualise()
        