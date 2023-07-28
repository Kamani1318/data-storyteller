# Import necessary libraries
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
# Machine Learning 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import GammaRegressor

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, completeness_score, homogeneity_score

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Custom classes 
from .utils import isNumerical
import os

def app():

    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        data = pd.read_csv('data/main_data.csv')
        copy_data = pd.read_csv("data/duplicate_main_data.csv")
        # Create the model parameters dictionary 
        params = {}

        # Use two column technique 
        col1, col2 = st.columns(2)

        # Design column 1 
        y_var = col1.radio("Select the variable to be predicted (y)", options=data.columns)

        # Design column 2 
        X_var = col2.multiselect("Select the variables to be used for prediction (X)", options=data.columns)

        # Check if len of x is not zero 
        if len(X_var) == 0:
            st.error("You have to put in some X variable and it cannot be left empty.")
            return

        # Check if y not in X 
        if y_var in X_var:
            st.error("Warning! Y variable cannot be present in your X-variable.")

        # Option to select predition type 
        pred_type = st.radio("Select the type of process you want to run.", 
                            options=["Regression", "Classification", "Clustering"], 
                            help="Write about reg and classification and clustering")

        # Add to model parameters 
        params = {
                'X': X_var,
                'y': y_var, 
                'pred_type': pred_type,
        }

        # if st.button("Run Models"):

        st.write(f"**Variable to be predicted:** {y_var}")
        st.write(f"**Variables to be used for prediction:** {X_var}")
        
        # Divide the data into test and train set 
        X = data[X_var]
        y = data[y_var]
        y_denorm = copy_data[y_var]

        # Check if y needs to be encoded
        if not isNumerical(y):
            st.write(f"Please make sure you preprocess the {y_var} before moving forwarrd!")
        if not isNumerical(X):
            st.write(f"Please make sure you preprocess the {X_var} before moving forwarrd!")
        

        # Perform train test splits 
        st.markdown("#### Train Test Splitting")
        size = st.slider("Percentage of value division",
                            min_value=0.1, 
                            max_value=0.9, 
                            step = 0.01, 
                            value=0.8, 
                            help="This is the value which will be used to divide the data for training and testing. Default = 80%")



        # Calculate the index to split the data
        split_index = int(size * len(X))

        # Split the data
        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]
        y_train_denorm = y_denorm[:split_index]
        y_test_denorm = y_denorm[split_index:]
        st.write("Number of training samples:", X_train.shape[0])
        st.write("Number of testing samples:", X_test.shape[0])

        # Save the model params as a json file
        with open('data/metadata/model_params.json', 'w') as json_file:
            json.dump(params, json_file)

        ''' RUNNING THE MACHINE LEARNING MODELS '''
        if pred_type == "Regression":
            st.write("Running Regression Models on Sample")

            # Linear regression model 
            linear_regression = LinearRegression()
            ridge_regression = Ridge()
            lasso_regression = Lasso()
            elastic_net_regression = ElasticNet()
            bayesian_ridge_regression = BayesianRidge()
            huber_regression = HuberRegressor()
            sgd_regression = SGDRegressor()
            passive_aggressive_regression = PassiveAggressiveRegressor()
            ard_regression = ARDRegression()
            tweedie_regression = TweedieRegressor()
            poisson_regression = PoissonRegressor()
            
            results_df_r = pd.DataFrame(columns=["Model", "R^2 Score", "MSE", "MAPE", "RMSE"])
            regression_models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "ElasticNet Regression": ElasticNet(),
            "Bayesian Ridge Regression": BayesianRidge(),
            "Huber Regression": HuberRegressor(),
            "SGD Regression": SGDRegressor(),
            "Passive Aggressive Regression": PassiveAggressiveRegressor(),
            "ARD Regression": ARDRegression(),
            "Tweedie Regression": TweedieRegressor(),
            "Poisson Regression": PoissonRegressor()
            }
            selected_models = st.multiselect("Select Regression Models", list(regression_models.keys()))
            results_df_r = pd.DataFrame(columns=["Model", "R^2 Score", "MSE", "MAPE", "RMSE"])
            with st.spinner("Modeling..."):
                for model_name in selected_models:
                    model = regression_models[model_name]  # Get the selected model
                    model.fit(X_train, y_train)  # Fit the model
                    y_pred = model.predict(X_test)  # Predict using the model

                    # Calculate evaluation metrics
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                    rmse = np.sqrt(mse)

                    # Append results to the output dataframe
                    new_row = pd.Series([model_name, r2, mse, mape, rmse], index=results_df_r.columns)
                    results_df_r = pd.concat([results_df_r, new_row.to_frame().T], ignore_index=True)
            st.dataframe(results_df_r)
                
        
        if pred_type == "Classification":
            if len(y_train_denorm) == len(y_train):
                st.write("Go ahead")
            else:
                st.error("Length of y denorm and y are not same")
            print(y_train_denorm)
            st.write("Running Classfication Models on Sample")

            # Create a list of all the models
            classification_models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Support Vector Machine": SVC(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Gaussian Naive Bayes": GaussianNB(),
                "Multinomial Naive Bayes": MultinomialNB(),
                "Multi-Layer Perceptron": MLPClassifier()
            }

            # Create a multiselect box for selecting classification models
            selected_models = st.multiselect("Select Classification Models", list(classification_models.keys()))

            # Create an empty DataFrame to store the evaluation scores
            results_df_c = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
                # Perform model fitting, prediction, and evaluation for selected models
            with st.spinner("Modeling..."):
                for model_name in selected_models:
                    model = classification_models[model_name]
                    model.fit(X_train, y_train_denorm)  # Fit the model
                    y_pred = model.predict(X_test)  # Predict using the model
                    print(y_pred)
                    print(y_test_denorm)
                    # Calculate evaluation metrics
                    accuracy = accuracy_score(y_test_denorm, y_pred)
                    precision = precision_score(y_test_denorm, y_pred, average = "micro")
                    recall = recall_score(y_test_denorm, y_pred, average = "micro")
                    f1 = f1_score(y_test_denorm, y_pred, average = "micro")

                    new_row = pd.Series([model_name, accuracy, precision, recall, f1], index=results_df_c.columns)
                    results_df_c = pd.concat([results_df_c, new_row.to_frame().T], ignore_index=True)

                # Display the results DataFrame
            st.dataframe(results_df_c)
                
        if pred_type=="Clustering" :
            ssd = []
            range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
            for num_clusters in range_n_clusters:
                kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
                kmeans.fit(X)

                ssd.append(kmeans.inertia_)

            # Plot the SSDs for each n_clusters
            fig, ax = plt.subplots()
            ax.plot(range_n_clusters, ssd, marker="o")
            ax.set_xlabel("Number of Clusters (n_clusters)")
            ax.set_ylabel("Sum of Squared Distances (SSD)")
            ax.set_title("Elbow Method - SSD vs n_clusters")
            
            n_cl = st.slider("Number of Clusters",
                            min_value=2, 
                            max_value=30, 
                            step = 1 )
            max_iter = st.slider("Maximum iterations" ,min_value = 10, max_value = 400, step = 10)
            kmeans = KMeans(n_clusters=n_cl, max_iter=max_iter)
            kmeans.fit(X)
            clus_labels = pd.DataFrame(kmeans.labels_, columns = ['Cluster Labels'])
            result = pd.concat([X,clus_labels], axis=1)
            
            # Display the plot in Streamlit
            st.pyplot(fig)
            st.dataframe(result)