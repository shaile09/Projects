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

# Presentation link - https://docs.google.com/presentation/d/1RmP25RHScKQilyfSACAFbge-9NbVzzcVNlpxeGMc4zk/edit#slide=id.p


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

# Link to the ERD Model  
![]( https://github.com/ebskii52/finalProject2020/blob/knar/segment%232/ERD_schema.PNG)

# Machine Learning Model

## Restaurant Entrepreneur: Predicting best type and location with Yelp Review Data
Bill (an investor) went on a work trip recently to Las Vegas and Phoenix for a few days. During his stay, he really liked the restaurant options that were available. Being that he already had interests in opening up a restaurant for some time now, he wanted to know what the type of restaurant and where to open would be most successful.
During Bill’s work trip, he used Yelp to search for different restaurants near his hotels. Bill is thinking about opening a restaurant in either Las Vegas or Phoenix as there is a lot of variations of food. Before he opens a restaurant, he is interested in learning more what types of restaurants are in these cities, what are the yelp stars, and additional details that would help him make a decision where to open restaurant.

As a first look, our group decided to use Yelp API to collect data on the mix of restaurants across cities and states within the United States. In this case, we decided to use the review data that Yelp is known for. Since Yelp provides a wide variety of restaurant categories, it is possible to get more information in regards to reviews and ratings for all different types in each city. After our initial attempts to use Yelp API, we found issues with importing the Yelp API data. When we created a dataframe from the API, it would only pull a small portion of the data on each run, not a sufficient amount of data to do a full analysis on. Fortunately, we were able to find some recent Yelp Review data on Kaggle.com that had a huge dataset to work with. The dataset included detailed information of the business such as, restaurant ID, location, postal codes, star ratings, review count, etc.

Before deciding on which Machine Learning Model we were going to choose to do our analysis on, we wanted to test a few to see what would be the best. We began setting up our code by importing various libraries, which included Random Forest Classifier and DeepLearning Machine Learning Models. Random Forest Classifier is a good model if you want high performance with less need for interpretation. Deep Learning Model is known for it's supremacy in terms of accuracy when trained with huge amounts of data and to get more neural network predictions. We've also imported train_test_split which will help us split our data for training and testing.
For the preprocessing, we imported StandardScaler and get_dummies. The StandardScaler is needed to transform the data so that it has a mean of 0 and a standard deviation of 1. The get_dummies is needed as it creates a binary column for each category type of restaurant.

## Why we chose these specific models and how do they work specifically with our dataset?
-	We chose Deep Learning, Random Forest Classifier and Logistic Regression as our Machine Learning Models. Random Forest is a good model for high performance with less need for interpretation. Deep Learning is known for it’s supremacy in terms of accuracy when trained with huge amounts of data and to get more neural network predictions. Logistic Regression is most useful when we want to predict the probability for a categorical response variable with two outcomes. In our case, we’re trying to decide on what category type of restaurant and where is best to open based off of reviews. A good review would be any scores between 3-5, bad would be 0-2.
## Detailed description of preliminary data preprocessing
-	We downloaded our data from Kaggle.com which was a json file and cleaned it. The cleaning process was filtering on only the data we needed, so we dropped a bunch of data, such as restaurants that were already closed or rows with null values. We filtered out all restaurant businesses based on category column into a new dataframe. Because we wanted to know what the best category type of restaurant to open was, we explored to see how many unique categories of restaurants were in the dataset, then created a new column called “ethnic_type” to put all the categories needed for our analysis. This new column is a feature we needed to add, to separate all the options we had available to decide on. We chose 20 unique restaurant types which were African, American, Asian_Fusion, British, Chinese, French, Greek, Hawaiian, Indian, Italian, Japanese, Korean, Mediterranean, Mexican, Middle_Eastern, Spanish, Thai, and Vietnamese. We did a value count for each city to see if there was enough data for Arizona and Nevada, which showed to have more than 1500 businesses for Las Vegas and Phoenix. This confirmed that we had enough data to conduct our analysis.  With this cleaned data, we created a new dataframe and then uploaded to postgres with a connection through SQL. In addition, we created CSV files of the cleaned data.
-	Further in the analysis process, we then added another column, which showed a prediction of star ratings, which were the review ratings on the graded scale of 0-5 stars on how satisfied customers were for each restaurant/business.
## Description of preliminary feature engineering and preliminary feature selection, including their decision making process
-	Initially, we started with Linear Regression Model and soon realized that we can only use this model when there are continuous values to be predicted. For categorical data prediction, Logistics Regression Model should be used.
-	Our goal is to predict the star rating, hence our attempts to find the models that could fit our output as well. While analyzing each Machine Learning model that we have tested, our Random Forest and Deep Learning samples seemed to be closer to our requirements.
How was the data split into training and testing sets?
The training and testing datasets were divided with 67% going towards training and 33% going towards testing. 
What was our model’s accuracy? What were the limitations and benefits of each model we chose?
## Deep Learning
Benefits – One main advantage is it’s capacity to execute feature engineering on it’s own. A deep learning algorithm will scan the data to search for features that correlate and combine them to enable faster learning without being explicitly told to do so. Another advantage is they produce the best results with unstructured data. Most company’s data is unstructured because of the different formats they all come in from. Unstructured data is hard to analyze for most machine learning models. Deep learning algorithms can be trained using different data formats, and still deliver good insight that’s relevant to the purpose of it’s training.
Limitations – it needs a large dataset to go through to predict the best outcomes, just like the human brain needs a lot of experiences to learn and deduce information before making any decisions. Overfitting is also another negative for the Deep Learning Model as it can train the data too well. Overtraining is a problem in neural networks. You can tell when a model is overtrained when the accuracy % stops improving after a certain number of epochs and flattens out.
For the dataset that we are using with ethnic type and city as our categorical data, our accuracy so far is around 20%. The challenge that we think is that we are trying to train the entire dataset just with category and number of reviews per rating. This makes the dataset small per category for it to be trained.
## Random Forest Classifier
Benefits – There is very little pre-processing that needs to be done. The data usually does not need to be rescaled or transformed. Predictions and training speeds are much quicker.
Limitations – For large datasets, they take up a lot of memory. They also tend to overfit.
For the dataset that we are using with ethnic type and city as our categorical data, our accuracy so far is around 20%. The challenge that we think is that we are trying to train the entire dataset just with category and number of reviews per rating. This makes the dataset limited per category for it to be trained.
## Logistic Regression
Benefits – It is easier to implement, interpret and very efficient to train. It gives an easy measure of how relevant a predictor is and it’s direction of association (positive or negative).
Limitations – It cannot solve non-linear problems.  It heavily relies on a proper presentation of your data. This means that logistic regression is not a useful tool unless you have already identified all the important independent variables. Since its outcome is discrete, Logistic Regression can only predict a categorical outcome. It is also an Algorithm that is known for its vulnerability to overfitting.
The accuracy rate in our analysis using this model is still 30%, which shows that the data is underfitting.


#### Description of the Tools 
To support our investor (Bill), we will be using Tableau to complete an analysis of the cleaned data sets of business and reviews datasets. As Bill is thinking about opening a restaurant, we need to provide him a tool that is interactive and very easy to use. To start, we will provide him with the number of distinct businesses that are in Phoenix and Las Vegas. Using the two datasets, we will complete the following analysis:
- Star rating of restaurants 
- Category of restaurants that are available 
- Comparison of the total review by type and postal code 
- City and postal code star rating 
- Prediction of stars by postal code 
- Prediction of future reviews by stars 
- Other resources from the web to help Bill determine where to open a resturant 
Using the Tableau platform, Bill will be able to interact with the dashboard and filter by postal code and category. In addition, we will create and style worksheets, dashboards, and stories, use worksheets to display data in a professional way, portray data accurately using dashboards.

#### Interactive Element of Tableau 
Tableau desktop provides analysts to create an interactive dashboard to allow users to search by various elements. The interactive element will help Bill to look at the data in various ways by filtering searching by various elements. This includes searching by which restaurants have the highest stars, categories that have highest stars, postal codes that have the highest stars, etc.




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

