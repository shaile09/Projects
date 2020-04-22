# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]
# # Yelp Data Prepocessing - ETL

# %%
# get_ipython().run_line_magic('pylab', 'inline')
# pd.set_option('display.max_columns',None)
# pd.options.display.max_seq_items = 2000
# pd.set_option('display.height', 1000)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# from gensim.corpora.dictionary import Dictionary
# from gensim.models.tfidfmodel import TfidfModel
# from wordcloud import WordCloud

from IPython import get_ipython
import seaborn as sns
import requests, re
import pandas as pd
import seaborn as sns
import os
import nltk
import string, itertools
from collections import Counter, defaultdict
from nltk.text import Text
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.cluster import KMeans
import json
from pandas.io.json import json_normalize
from sqlalchemy import create_engine
from config import db_password


# %% [markdown]
#   ### Yelp businesses Prepocessing

# %%
# Open JSON file
with open(f'dataset/business.json', mode='r', encoding="utf8") as file:
    yelp_source = [json.loads(line) for line in file]

# %%
# Put JSON file into a Dataframe
business_df = pd.DataFrame(yelp_source)
business_df.head()

# %%
# Remove all closed restaurants
a = [0]
business_df = business_df[~business_df['is_open'].isin(a)]

# %%
## remove quotation marks in name and address column
business_df.drop(['neighborhood', 'is_open', 'address'], axis=1, inplace=True)
business_df.head()

# %%
# Finding the state that has most values in the dataset
business_df['state'].value_counts()

# %%
# Filter all businesses of AZ
states = ["AZ", "NV"]

# Filter all businesses of AZ & 
##states = ["AZ"]
aznv_df = business_df.loc[business_df['state'].isin(states)]
aznv_df.head()

# %%
aznv_df = aznv_df.copy()

# %%
## Cerate new column with categories
aznv_df['category'] = aznv_df['categories'].apply(lambda x: ','.join(map(str, x)))
aznv_df.head()

# %%
# Filter all the restaurants businesses
AZNV_restaurants = aznv_df[aznv_df['category'].str.contains('Restaurants')]
AZNV_restaurants.head()

# %%
# see all the unique values for category
column_values = AZNV_restaurants[["category"]].values.ravel()
unique_values =  pd.unique(column_values)
print(unique_values)

# %%
AZNV_restaurants = AZNV_restaurants.copy()

# %%
# Filtering out type of Restaurants
AZNV_restaurants.loc[AZNV_restaurants.category.str.contains('American'),'EthnicType'] = 'American'
AZNV_restaurants.loc[AZNV_restaurants.category.str.contains('Mexican'), 'EthnicType'] = 'Mexican'
AZNV_restaurants.loc[AZNV_restaurants.category.str.contains('Italian'), 'EthnicType'] = 'Italian'
AZNV_restaurants.loc[AZNV_restaurants.category.str.contains('Japanese'), 'EthnicType'] = 'Japanese'
AZNV_restaurants.loc[AZNV_restaurants.category.str.contains('Chinese'), 'EthnicType'] = 'Chinese'
AZNV_restaurants.loc[AZNV_restaurants.category.str.contains('Thai'), 'EthnicType'] = 'Thai'
AZNV_restaurants.loc[AZNV_restaurants.category.str.contains('Mediterranean'), 'EthnicType'] = 'Mediterranean'
AZNV_restaurants.loc[AZNV_restaurants.category.str.contains('French'), 'EthnicType'] = 'French'
AZNV_restaurants.loc[AZNV_restaurants.category.str.contains('Vietnamese'), 'EthnicType'] = 'Vietnamese'
AZNV_restaurants.loc[AZNV_restaurants.category.str.contains('Greek'),'EthnicType'] = 'Greek'
AZNV_restaurants.loc[AZNV_restaurants.category.str.contains('Indian'),'EthnicType'] = 'Indian'
AZNV_restaurants.loc[AZNV_restaurants.category.str.contains('Korean'),'EthnicType'] = 'Korean'
AZNV_restaurants.loc[AZNV_restaurants.category.str.contains('Hawaiian'),'EthnicType'] = 'Hawaiian'
AZNV_restaurants.loc[AZNV_restaurants.category.str.contains('African'),'EthnicType'] = 'African'
AZNV_restaurants.loc[AZNV_restaurants.category.str.contains('Spanish'),'EthnicType'] = 'Spanish'
AZNV_restaurants.loc[AZNV_restaurants.category.str.contains('Middle_Eastern'),'EthnicType'] = 'Middle_Eastern'
AZNV_restaurants.loc[AZNV_restaurants.category.str.contains('Asian Fusion'),'EthnicType'] = 'Asian_Fusion'
AZNV_restaurants.loc[AZNV_restaurants.category.str.contains('British'),'EthnicType'] = 'British'

AZNV_restaurants.head()

# %%
# Dropping all rows that are null - as this will skew the data
AZNV_restaurants.dropna(inplace=True)

# %%
# Remove category and categories columns
AZNV_restaurants.drop(['category', 'categories'], axis=1, inplace=True)
AZNV_restaurants.head()

# %%
# Reset the index
AZNV_restaurants.reset_index(inplace = True) 

# %%
AZNV_restaurants.count()

# %%
#length of unique values for city
len(AZNV_restaurants['city'].unique())

# %%
# Count unique city values
AZNV_restaurants['city'].value_counts()

# %%
#Filter values for only Phoenix and Las Vegas
city = ['Phoenix', 'Las Vegas']
AZNV_restaurants = AZNV_restaurants[AZNV_restaurants['city'].isin(city)]
AZNV_restaurants.head(n=20)

# %%
# Create businesses dataframe to import to posgres
businesses = AZNV_restaurants.filter(['business_id', 'name'], axis=1)
businesses.head()

# %%
# Create business_info dataframe to import to posgres
business_info = AZNV_restaurants.filter(['business_id', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'review_count', 'EthnicType', 'stars'], axis=1)
business_info.head()

# %% [markdown]
# ### Business Review Prepocessing

# %%
#Import Business review JSON file
file_dir = 'C:/Users/knush/GitRepository/Class_Practice/FinalProject/dataset/'
f'{file_dir}review.json'

# %%
# Open business JSON file
with open(f'{file_dir}review.json', mode='r', encoding="utf8", errors='ignore') as f:
    reviews_data = [json.loads(line) for line in f]

# %%
# Import to dataframe
reviews_df = pd.DataFrame(reviews_data)
reviews_df.head()

# %%
#Count the rows
reviews_df.count()

# %%
# Merge/Combine the reveiw_df and us_restaurants datas into a single dataset.
business_reviews = pd.merge(left=business_info, right=reviews_df, how='left', left_on='business_id', right_on='business_id')
business_reviews.head()

# %%
# Row counts
business_reviews.count()

# %%
# drop null values
business_reviews.dropna()

# %%
#rename start columns
business_reviews.rename(columns={"stars_x": "restaurant_star", "stars_y": "review_star"}, inplace = True)
business_reviews.head()

# %%
# Row counts
business_reviews.count()

# %%
# Category counts
categoryCounts=business_reviews.EthnicType.value_counts()
categoryCounts

# %%
business_reviews.columns

# %%
# Create a dataframe needed for Machine Learning model to import to posgres
mlbusiness_reviews = business_reviews.filter(['review_id', 'business_id' , 'review_star', 'useful', 'EthnicType', 'city', 'state', 'postal_code', 
'latitude', 'longitude',], axis=1)
mlbusiness_reviews.head()

# %%
mlbusiness_reviews.count()

# %%
# Create business_reviews dataframe to import to posgres
business_reviews_df = business_reviews.filter(['review_id', 'user_id', 'business_id', 'date', 'review_star', 'text', 'useful', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'EthnicType'], axis=1)
business_reviews_df.head()

#%%
# # Make a connection with PostgreSQL - pull the SQL password from config.py file
"postgres://[user]:[password]@[location]:[port]/[database]"
# # Make a connection with PostgreSQL database
db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/Yelp_db"
# # Create the database engine with the following
engine = create_engine(db_string)
# # Import the Yelp data to SQL

#businesses > businesses table
businesses.to_sql(name='businesses', con=engine, if_exists='replace' ,index=False)

#business_info > business_info table
business_info.to_sql(name='business_info', con=engine, if_exists='replace' ,index=False)


#business_reviews_df > business_reviews table
business_reviews_df.to_sql(name='business_reviews', con=engine, if_exists='replace' ,index=False)

#mlbusiness_reviews > mlbusiness_reviews table
mlbusiness_reviews.to_sql(name='mlbusiness_reviews', con=engine, if_exists='replace' ,index=False)


# %%
# Export table to csv file if needed

#businesses > businesses.csv
businesses.to_csv('businesses.csv')

#business_info > business_info.csv
business_info.to_csv('business_info.csv')

#business_reviews_df > business_reviews.csv
business_reviews_df.to_csv('business_reviews_df.csv')

#mlbusiness_reviews > mlbusiness_reviews.csv
mlbusiness_reviews.to_csv('mlbusiness_reviews.csv')

# %%
