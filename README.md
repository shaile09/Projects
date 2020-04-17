# Final Project 


# Team's Communication Protocols

 - Our team mainly communicates and will communicate through Slack and Zoom meetings. 
 - Will meet during the week to work on the project either whole group or small groups other than classtime
 - Will also work individually to complete certain aspects of the project.
 - Segment 1 
          - Met several times (more than two times) outside class time to work on the project.
          - Worked on choosing the topic for the project and finding the dataset. 
          - Worked on import the dataset into a pandas dataframe.
          - Worked on deciding what technologies to use.
 - Segment 2
         - TBA
          

=======

## First Segment 

This week's project focused on the selecting a project idea. The goal of this week was to decide on our overall project, selecting your question, building a model, finding a dataset, using a database using CSV or JSON files to prototype our idea. In this segment, we began by gathering a project team to help support the project.

### Background

As part of this project, we are woking with Bill, who is an investor in the west cost. Bill recently went to visit San Diego for a few days and he really liked the restaurant options that are available since he is interested in opening a restaurant business himself.  
In the few days he was in San Diego, he was using the Yelp to choose different restaurants near his hotel. Bill is thinking about opening a restaurant in San Diego as there is a lot of variations of food. Before he opens a restaurant, he wants to learn about the   about what types of restaurants are in San Diego, what are the Yelp review ratings (good or bad), what is the rating of the new resturants, and additional details that would help him open a restaurant. 

### Project Team 

Prior to starting the project, we put together a list of team members to support the project. 

1) Square - Responsible for setting up the repository. This includes naming the repository and adding team members (branches).
Primary: Ebrahim, Seghen

2) Triangle - Responsible for creating a simple machine learning model with these questions in mind. This can also be a diagram that explains how it will work concurrently with the rest of the project steps. Team members will focus on which model to use, methods to use on training the data for the model, and level of accuracy for the model.
Primary: Lisa, Knar, Seghen

3) Circle - Responsible for the mockup database with a set of sample data, or even fabricated data. This will ensure the database will work seamlessly with the rest of the project. This includes a document that describes our schema of the database, which can be a markdown document or ERD. 
Primary: Knar, Justin, Lisa

4) X - Focuses on what technologies will be used for each section of the project. 
Primary: Justin, Knar, Ebrahim

### Description of the source of data 

The project team is planning to use a dataset that is extracted from the Kaggle website. 
Business Details: https://www.kaggle.com/yelp-dataset/yelp-dataset#yelp_academic_dataset_business.json
| business_id  |
|--------------|
| name         |
| address      |
| city         |
| state        |
| postal_code  |
| latitude     |
| longitude    |
| stars        |
| review_count |
| is_open      |
| attributes   |
| categories   |
| hours        |

### Schema

Preliminary schema of the database is inlcuded as part of this segment. We are intending to use PgAdmin and Postgres.

### Hypothesis 

Null Hypothesis: There is no correlation between the type of restaurant and restaurant review ratings in the city of San Diego.
Hypothesis: There is a correlation between the type of restaurant and restaurant review ratings in the city of San Diego.
Dataset = Yelp or restaurant reviews
Label = rating

### Analysis Questions



# Project Overview

- Bill (an investor) went to work to visit San Diego for a few days and he really liked the restaurant options that are available since he is interested in opening a restaurant business himself.  

-In the few days he was in San Diego, he was using the Yelp to choose different restaurants near his hotel. Bill is thinking about opening a restaurant in San Diego as there is a lot of variations of food. Before he opens a restaurant, he is interested in learning more about what types of restaurants are in San Diego, what are the yelp review ratings, and additional details that would help him open a restaurant. 


# Description of the source of data

 - The team is planning to connect to the Yelp API to conduct this analysis.
 - Business Details: https://www.yelp.com/developers/documentation/v3/business
 - Detailed information of the businesses (restaurant id, location, address, phone number, price etc.) 
 - Business reviews: https://www.yelp.com/developers/documentation/v3/business_reviews
 - Includes restaurant id, rating, user id, time created, text, etc.


Null Hypothesis: There is no correlation between the type of restaurant type of restaurant NYC and review restaurant ratings  
Hypothesis: There is a correlation between the type of restaurant type of restaurant NYC and review restaurant ratings 
Dataset = Yelp or restaurant reviews
Label = rating


## Analysis Questions

We are hoping to answer the following questions:

- What types of restaurants are in San Diego?
- What are the Yelp review ratings (good or bad)?
- What is the rating of the new resturants?
- What other information does he need to open a restaurant?
- Predictive of the restaurant ratings (good or bad) 
- Predict future reviews for a new restaurant based on the available data
- Predict what a reviewer will review for a specific data
- Predict what restaurants may close in near future and what will be an optimal business at that location

- Predict what restaurant to open where

### Machine Learning Model 

The project team is planning to use a linear regression model to determine if the rating for the new resturants will be good or bad. The data for star rating is continous and we belive that we could use a linear regression model to determine if it will be good or bad. 
Good star rating = ratings >= 4
Bad star rating = ratings <= 3

### Technology to be used 

#### Data Cleaning and Analysis
We will be using Pandas to clean the data and perform exploratory analysis. Further analysis will be conducted using Python. We will be importing the yelp data downloaded from kaggle.com, and converting that dataset, which is in JSON format, into a dataframe.

#### Database Storage
Postgres is the database we will be using.

#### Machine Learning
Linear Regression, Deep Learning, and Random Forest Classifier are the machine learning models we will be using to train and test our data. We will then choose which machine learning model is the best of the 3.

### Dashboard

Tableau is the application we will be using to display our data. We feel that tableau will give the user the most simplest way of looking at which location geographically is best when considering location and type of restaurant to open. It will include comparable data, such as what type of restaurant it is, whether it got a good or a bad rating, and exactly where it is located.



- Predict what type of restaurant to open and where based off of review ratings (Bad 0-2, Good 3-5)
- Users will be able to see where are the good restaurants vs. not

=======
- Predict what restaurant to open where


