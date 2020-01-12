# Disaster Response Pipeline Project

##About
This project is part of Udacity Data Science Nanodegree. The project's goal is to build a pipeline to categorize messages sent during natural disasters into one or more of the 36 available categories using Natural Language Processing tools. The data is provided by [Figure Eight](https://www.figure-eight.com/dataset/combined-disaster-response-data/)

## Requirements

This project uses the following Python libraries

* `NumPy` 
* `pandas`
* `matplotlib`
* `scikit-learn`

## Folder structure 
* Data 
  - `process_data.py` : This file performs ETL (extract, transform, and load) tasks 
* Model 
  - `train_classifier.py` : This file creates a machine learning pipeline and return a multi-label classification model
* App
  - `run.py` : Starts up local server and serve a web app where you can use the machine learning model to categorize a user input message


## Usage
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
