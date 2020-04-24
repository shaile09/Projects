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
         - Met on Tuesdays, Thursdays and Saturdays as a group.
	 - Communicated through Slack group chat and Zoom meetings outside class time.
	 - Worked on the preprocessing the dataset for Machine Learing Model and Visualazation.
	 - Worked on creating Machine Learning Model.
	 - Worked on creating Google Slides for the project description.
	 - Developed a protype of the data anytics dashaboard (Tabluea)
	
=======
# Project Overview

 - Bill (an investor) went to work to visit Phoenix and Las Vegas for a few days and he really liked the restaurant options that are available since he is interested in opening a restaurant business himself in either Phoenix or Las Vegas .  
 - In the few days he was in both cities, he was using the Yelp to choose different restaurants near his hotels. Bill is thinking about opening a restaurant in either Las Vegas or Phoenix as there is a lot of variations of food. Before he opens a restaurant, he is interested in learning more what types of restaurants are in these cities, what are the yelp stars, and additional details that would help him make a decision where to open restaurant.
 
# Add images 

## Description of the source of data

 - The team found a dataset on Kaggle.com to conduct this analysis - https://www.kaggle.com/yelp-dataset/yelp-dataset/version/2
 - From the data set will be using business data
 	- Detailed information of the businesses (restaurant id, location, postal code, stars, review count, etc.) 
 - And will also use Business reviews data
 	- Includes restaurant id, stars, user id, text, etc.

=======
# Hypothesis

Null Hypothesis: There is no correlation between the type of restaurant and review restaurant star rating based on the either Las Vegas or Phoenix
Hypothesis: There is a correlation between the type of restaurant and review restaurant ratings based on the either Las Vegas or Phoenix
Label = stars

=======
# Analysis Questions

We are hoping to answer the following questions:

- What types of restaurants are in Phoenix and Las Vegas? To choose the mail restaurant types for the analysis
- What are the Yelp review overall ratings?
- What is the rating of the resturants postal code and category?
- What other information does the investor need to open a restaurant?
- Predictive of the restaurant stars based on postal code
- Predict future reviews for a new restaurant based on the available data
- Predict what restaurant to open where

# Second Segment

# Placeholder Presentation link - 


## ETL / Prepocessing of the Yelp Data.

- Import and analyze yelp businesses dataset.
	- Opened business.json file and imported into a dataframe.
	- The Yelp dataset includes data on different businesses other than restaurants.
	- Filtered only open businesses and dropped columns that will not be used for the analysis (neighborhood, is_open).
	- Counted how many values each state has:
		- The result showed that Arizona and Nevada states have the most values, which indicates that we have enough data to do our analysis.
	- Filtered out all the restaurant businesses based on category column into new dataframe.
	- Explored to see how many unique categories of restaurants are in the dataset
		- Created a new column, ethnic_type, to put all the categories needed for the analysis.
		- Chose only 20 unique restaurant types:
			- African, American, Asian_Fusion, British, Chinese, French, Greek, Hawaiian, Indian, Italian, Japanese, Korean, Mediterranean, Mexican, Middle_Eastern, Spanish, Thai, Vietnamese
		- Explored the value count for the cities to see if we have enough data to do analysis for cities Phoenix and Las Vegas.
			- Data shows we have 2455 business for Las Vegas and 1557 businesses for Phoenix.
			- Dropped all the data except for Phoenix and Las Vegas.
			- Created business_info dataframe to import the tables to posgres.
- Import and analyze yelp reviews dataset.
	- Opened review.json file and put the data into reviews_df dataframe.
	- Merged reviews_df and business_info dataframes by business_id to select all the reviews that are matched with are selected businesses.
	- Dropped rows with null values.
	- Rename star_x column to restaurant_star and star_y to review_star as star_x is the restaurant rating from business table and star_y is the reviewers rating for the corresponding restaurant.
	- Created business_reviews_df dataframe to import to posgres.
	- Created a connection to SQL and imported the tables of the cleaned data into posgres.
	- In addition, created csv files of the cleaned data.

- Database
	- Database stores business_info, business_reviews and review_prediction (machine_learning) tables.
	- Yelp_db is used to pull from the database the business_reviews and business_info tables for machine learning.
	- The  business_info and business_reviews are joined to create join within posgres. # Placeholder 
	- Due to restrictions on the cost of AWS, we do not have shared databased, instead use local posgres database with same table schema (schema.sql) to acess the data.
		- The sample database can be accessed at https://yelpdbbackup.s3.us-east-2.amazonaws.com/Yelp_db.sql. 

# Placeholder Add ERD 

# Placeholder Machine Learning Model

# Placeholder Dashboard 

# First Segment 

This week's project focused on the selecting a project idea. The goal of this week was to decide on our overall project, selecting your question, building a model, finding a dataset, using a database using CSV or JSON files to prototype our idea. In this segment, we began by gathering a project team to help support the project.

### Background

 - Bill (an investor) went to work to visit Phoenix and Las Vegas for a few days and he really liked the restaurant options that are available since he is interested in opening a restaurant business himself.  
 - In the few days he was in both cities, he was using the Yelp to choose different restaurants near his hotels. Bill is thinking about opening a restaurant in either Las Vegas or Phoenix as there is a lot of variations of food. Before he opens a restaurant, he is interested in learning more what types of restaurants are in these cities, what are the yelp reviews, and additional details that would help him make a decision where to open restaurant.

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
Business Details: https://www.kaggle.com/yelp-dataset/yelp-dataset#yelp_academic_dataset_business.json - preprocessed this dataset and determined that it does not have the data needed to be used for the anaysis
Other dataset used: https://www.kaggle.com/yelp-dataset/yelp-dataset/version/2 - used this dataset as it has the data for the cities we need to do the analysis
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
=======

