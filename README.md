# Final Project - Restaurant Types and Reviews 

=======
# Project Overview

 - Bill (an investor) went to work to visit Phoenix and Las Vegas for a few days and he really liked the restaurant options that are available since he is interested in opening a restaurant business himself in either Phoenix or Las Vegas .  
 - In the few days he was in both cities, he was using the Yelp to choose different restaurants near his hotels. Bill is thinking about opening a restaurant in either Las Vegas or Phoenix as there is a lot of variations of food. Before he opens a restaurant, he is interested in learning more what types of restaurants are in these cities, what are the yelp stars, and additional details that would help him make a decision where to open restaurant.
 
=======
# Description of the source of data

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

=======
# Project Outline
- All previous segment README submissions are in this ![Past README]( https://github.com/ebskii52/finalProject2020/blob/seghen/README%20from%20Segment%201-2-3.txt) file. We updated main README to reflect our final project.

- requirement.txt added to Github that includes the packages used.
- Softwares Used:
	-Python 3.6.9:: Anaconda, Inc., Jupyter Notebook, 6.0.2, Visual Studio Code, 1.40.2., PgAdmin 4, Tableau Desktop
- Data Cleaning and Analysis
	- Used Pandas to clean the data and perform exploratory analysis. Further analysis conducted using Python. 
	- We imported the yelp data downloaded from kaggle.com, and converted that dataset, which is in JSON format, into a dataframe. 
	- Imported JSON to read json file and used Pandas pd.DataFrame to turn the data into a dataframe
	- Used sqlalchemy create_engine to make a connection to posgres sql database
- Database Storage
	- Postgres is the database used to store clean data for analysis, machine learning model and prediction table for visualization.
	- Machine Learning
	- Logistic & Linear Regression, Deep Learning, and Random Forest are the machine learning models used to train and test our data. 
	- From sklearn library used different machine learning models to train and test our data. 
- Dashboard
	-Tableau is the application used to display our data. We feel that tableau will give the user the most simplest way of looking at which location geographically is best when considering location and type of restaurant to open. It will include comparable data, such as what type of restaurant it is, whether it got a good or a bad rating, and exactly where it is located.


## ![Presentation link]( https://docs.google.com/presentation/d/1RmP25RHScKQilyfSACAFbge-9NbVzzcVNlpxeGMc4zk/edit#slide=id.p)

## ETL / Prepocessing of the Yelp Data - ![Yelp_ETL.py]()

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
		
## Link to the ERD Model
![]( https://github.com/ebskii52/finalProject2020/blob/knar/segment%232/ERD_schema.PNG)
	
## Machine Learning Process - ![YelpRandomForest.py]( https://github.com/ebskii52/finalProject2020/blob/master/YelpRandomForest.py)
- Random Forest Classifier/Regressor - Little pre-processing needs to be done. The data usually does not need to be rescaled or transformed. Predictions and training speeds are much quicker. The main difference between the two is that the output variable in regression is numerical (or continuous) while that for classification is categorical (or discrete).
- Random Forest Regressor. Use average of all the branches to predict value. Used 1 to 5 as continuous values.
The accuracy here came out to be around 90%. These were identified by gridSearchCV.
Parameters used n_estimators=50, max_depth=32, random_state=2.
- The various Machine Learning model were tested and analized and details are in ![Past README]( https://github.com/ebskii52/finalProject2020/blob/seghen/README%20from%20Segment%201-2-3.txt) file.
- Considering the best accuracy rate achieved so far we decided to use Random Forest Regressor.

## Dashboard - Tableau 
![Link to Tableau Dashboard]( https://public.tableau.com/profile/seghen7339#!/vizhome/Workbook_FinalProject_test2/Story1?publish=yes)
	- Each Dashboard slide has an interactive component where the user can choose to display data either by city, postal_code or restaurant category (ethnic_type). 
	- Dashboard: 
		- shows the total business counts for Las Vegas (2455) and Phoenix (1557). 
		- shows that Las Vegas restaurants have higher rating that Phoenix as there are more restaurants in the area.
		- shows that Las Vegas and Pheonix have similar outcomes when compared by the ethnic type.
		- shows a tree map by postal code, where postal code 89109 has a higher ratings.
		- shows comparison between machine learning model prediction and the original dataset, where the results show similar conclusions. 
		- shows stars data from the original dataset by ethnic_type.
		- The results show that ethnic_type American has the most rating - it is also because our dataset includes more American restaurants that other ethnic_types.
		
### Dashboard Images

![]( https://github.com/ebskii52/finalProject2020/blob/knar/DashboardImage.PNG)

![]( https://github.com/ebskii52/finalProject2020/blob/knar/DashboardImage1.PNG)

### Dashboard and Analysis

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

## Results based on Analysis
- Overall,  postal code 89109 in Las Vegas has a high number of customers for ethnic type (American) foods.
- Based on our analysis,  American food types have a higher number star rating compared to any other ethnic type. 
- Our recommendation for our investor is to open an American restaurant in postal code 89109 since most customers favor American restaurants. 

## Limitations and Next Steps
- Our data included limited number of fields for analysts and we are not able to pin point address of of our restaurant. 
Dataset was too small to run any of the machine learning models. As a result, we were not able to include the features for our model.
The data visualized in Tableau is an extract of a csv dataset since we could not  use an  AWS server due to cost.
- Next Steps:
	- Find additional data points to be included in our model for a more accurate machine learning model.
	- Find additional datasets to combine our original data
