<!-- Add logo -->
<!--  ![App Logo](https://i.stack.imgur.com/ARgpq.jpg) -->

![data-storyteller](https://socialify.git.ci/prakharrathi25/data-storyteller/image?description=1&descriptionEditable=Automated%20tool%20for%20data%20analysis%2C%20visualization%2C%20feature%20selection%2C%20machine%20learning%20and%20inference%20in%20one%20application!&font=Bitter&forks=1&logo=https%3A%2F%2Fcamo.githubusercontent.com%2Fba46960c1170c1d56a4fcfdd375be6b13852795e31523ea76bde3366f021c25d%2F68747470733a2f2f692e737461636b2e696d6775722e636f6d2f41526770712e6a7067&owner=1&pattern=Floating%20Cogs&stargazers=1&theme=Light)

[![forthebadge](https://forthebadge.com/images/badges/built-by-developers.svg)](https://forthebadge.com)
[![pythonbadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

# 📱 Data Storyteller 📉

_**ONE STOP SOLUTION FOR ALL YOUR DATA NEEDS**_ 
## Introduction 

As per Gartner [2], the analytics and business intelligence platform market has transitioned from the visual data discovery era to the augmented era. Data and analytics leaders/administrators should begin piloting capabilities and competencies that enable the “augmented consumer”.

With the technology advancements, the organisation today has the pre-eminence of taking data driven decisions and strategize their planning, forecasts based on the same. A profusion of business users do not have time to analyze the data and then secure noteworthy insights. And there are gaps betwixt how the tool produces an output and how the business user can exploit it to interpret it. In accompaniment, it needs a admirable domain knowledge to build business insights from data. Not every user is a business expert. Given a snapshot of data, we would like to fabricate a system which can verbalise a story from the data. The story includes the automation in the sense of being driven by the data, context and personal preferences. In this case, it solves both the problems of the tool usage as well as guiding the user with the data driven intelligence to make business decision. The whole corollary is driven by outcome and effectiveness.

## Tool Description 

Data Storyteller is an AI based tool that can take a data set, identify patterns in the data, can interpret the result, and can then produce an output story that is understandable to a business user based on the context. It is able to pro-actively analyse data on behalf of users and generate smart feeds using natural language generation techniques which can then be consumed easily by business users with very less efforts. The application has been built keeping in mind a rather elementary user and is hence, easily usable and understandable. This also uses a **multipage implementation** of Streamlit Library using Class based pages. 

## Note: This is not the original repo but a different version that I have created. This repo was created for educational purposes only and no plagarism was intended. 
## Features 

Given data/analytics output, the tool can:-

- extract numerical and categorical columns of a given dataset. 
- convert wrongly classified categorical data to numeric.
- provide data vasualisation with 10 options of choosing the chart type.
- regression classification and clustering options to user to choose from and apply on the dataset.
- preprocessign option that automatically converts data into numeric format.

## 📝 Module-Wise Description

The application also uses Streamlit for a multiclass page implementation which can be viewed in the `multipage.py` file. The UI of the application can be seen here. The application is divided into multiple modules, each of which have been described below.

![UI of the application](https://i.stack.imgur.com/MOVpz.png)


_📌 **Data Upload**_ <br/>

This module deals with the data upload. It can take csv and excel files. As soon as the data is uploaded, it creates a copy of the data to ensure that we don't have to read the data multiple times. It also saves the columns and their data types along with displaying them for the user. This is used to upload and save the data and it's column types which will be further needed at a later stage. 

_📌 **Change Metadata**_ <br/>

Once the column types are saved in the metadata, we need to give the user the option to change the type. This is to ensure that the automatic column tagging can be overridden if the user wishes. For example a binary column with 0 and 1s can be tagged as numerical and the user might have to correct it. The three data types available are:

* Numerical 
* Categorical 
* Object

The correction happens immediately and is saved at that moment. 

_📌 **Machine Learning**_ <br/>

This section automates the process of machine learning by giving the user the option to select X and y variables and letting us do everything else. The user can specify which columns they need for machine learning and then select the type of process - regression, classficiation and clustering. The application selects multiple models and saves the best one as a binary `.sav` file to be used in the future for inferencing. 10 different models are available to use for regression and classification and K means is available for Clustering currently. On choosing 1 or more models the respective metrics will appear in a tabular format. 

_📌 **Data Visualization**_ <br/>
This section automates teh data visualisation process of data analysis. The user has an optiont to choose any two data columns that are present in the dataset alogn with the chart type. One choosing these three,, the graph will be displayed.

_📌 **Y-Parameter Optimization**_ <br/>

## Technology Stack 

1. Python 
2. Streamlit 
3. Pandas
4. Scikit-Learn
5. Seaborn

# How to Run 

- Clone the repository
- Setup Virtual environment
```
$ python3 -m venv env
```
- Activate the virtual environment
```
$ source env/bin/activate
```
- Install dependencies using
```
$ pip install -r requirements.txt
```
- Run Streamlit
```
$ streamlit run app.py
```

## Other Content

## 🤝 How to Contribute? [3]

- Take a look at the Existing Issues or create your own Issues!
- Wait for the Issue to be assigned to you after which you can start working on it.
- Fork the Repo and create a Branch for any Issue that you are working upon.
- Create a Pull Request which will be promptly reviewed and suggestions would be added to improve it.
- Add Screenshots to help us know what this Script is all about.


# 👨‍💻 Original Contributors ✨

<table>
  <tr>
    <td align="center"><a href="https://github.com/prakharrathi25"><img src="https://avatars.githubusercontent.com/u/38958532?v=4" width="100px;" alt=""/><br /><sub><b>Prakhar Rathi</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/mpLogics"><img src="https://avatars.githubusercontent.com/u/48443496?v=4" width="100px;" alt=""/><br /><sub><b>Manav Prabhakar</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/salilsaxena"><img src="https://avatars.githubusercontent.com/u/54006908?v=4" width="100px;" alt=""/><br /><sub><b>Salil Sxena</b></sub></a><br /></td> 
  </tr>
</table>


## References 



## Contact

For any feedback or queries, please reach out to [prakharrathi25@gmail.com](prakharrathi25@gmail.com).

Note: The project is only for education purposes, no plagiarism is intended.
