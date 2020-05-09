# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#Random Forest Regressor
# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
# Import library for Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
import os
from config import db_password
import sys

# %%
f= open("RandomForestLog.txt","w+")

# %%
# Make a connection with PostgreSQL - pull the SQL password from config.py file
"postgres://[user]:[password]@[location]:[port]/[database]"
# # Make a connection with PostgreSQL database
db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/Yelp_db"

# %% [markdown]
# Clean data is loaded into postgress. Based on all the data 
# inputs available, and trying mulitple models like Random Forest 
# Classifier, Random Forest Regressor, Deep learning, Linear Regression 
# and Logistic Regression, we came to conclusion that 
# Random Forest Regression fits the best with our data. 
# This improves our accuracy scores and does give resonable 
# weightage to all the features that we were using. 
# As of now I am getting accuracy as 91%

# %%
#Connection to postgres and pull data from tables loaded after 
#scrubing
try:
    # Create the database engine with the following
    engine = create_engine(db_string)
    from sqlalchemy.orm import Session
    session = Session(engine)
except psycopg2.DatabaseError as error:
    f.write("Database connectivity error")   
    f.close()
    sys.exit()

# %%
#Write a query to joining multiple tables from postgres database and 
# load the data into data frame
try:
    reviewsDF = pd.read_sql('select r.review_star stars,b.city, b.postal_code, \
        r.ethnic_type from business_reviews r, business_info b where \
            b.business_id = r.business_id and length(b.postal_code)>0',engine)
except psycopg2.DatabaseError as error:
    f.write("Error in query selection") 
    f.close()
    sys.exit()

# %%
#Divide data into X and y inputs for training the 
y = reviewsDF.stars
yDF=round(pd.DataFrame(y))
X = reviewsDF

# %%
#Combine/Binning ethnic types if review counts are less then 4000.
category_countsX=X.ethnic_type.value_counts()
replace_type=list(category_countsX[category_countsX<4000].index)


# %%
#Combine ethnic types with others
for application in replace_type:
    X.ethnic_type =  X.ethnic_type.replace(application,"Others")

# %%
# Generate our categorical variable list
review_catX = X.dtypes[X.dtypes == "object"].index.tolist()


# %%
#Remove decimal from star review
yDF['stars'] = yDF['stars'].astype(str).replace('\.0', '', regex=True)

# %%
# Generate our categorical variable list for y
review_caty = yDF.dtypes[yDF.dtypes == "object"].index.tolist()


# %%
predict_inputDF = X
predict_inputDF['stars'] = round(predict_inputDF['stars'])


# %%
#Start preparing reviews Dataframe for final output
reviews_for_output = pd.DataFrame(predict_inputDF.groupby(['postal_code',\
    'city','ethnic_type'],as_index=False).sum())
reviews_for_output = reviews_for_output.drop(['stars'], axis = 'columns')

# %%
XInput=predict_inputDF.drop(columns=['stars'])



# %%
#Start preparing reviews Dataframe for final output
reviews_for_output = pd.DataFrame(predict_inputDF.groupby(['postal_code','city','ethnic_type'],as_index=False).sum())
reviews_for_output = reviews_for_output.drop(['stars'], axis = 'columns')


# %%
XInput=predict_inputDF.drop(columns=['stars'])

# %%
#Explode categories. This is similar to oneHotEncoder
#Get a dataframe ready for predicting entire dataset for final output 
# prediction
dummy_categories = pd.get_dummies(XInput.ethnic_type)
dummy_city = pd.get_dummies(XInput.city)

# %%
#Get a dataframe ready for preparing training and testing datasets
new_review_all = pd.concat([XInput, dummy_categories], axis = 'columns')
new_review_all = pd.concat([new_review_all, dummy_city], axis = 'columns')
final_PC = new_review_all.drop(['city', 'ethnic_type'], axis = 'columns')
X=final_PC
X_test_data= X


# %%
#Getting output dataset ready. This means formating it the same as X but should have unique features only.
dummy_categories1 = pd.get_dummies(reviews_for_output.ethnic_type)
dummy_city1 = pd.get_dummies(reviews_for_output.city)
final_outputX = pd.concat([reviews_for_output, dummy_categories1], \
axis = 'columns')
final_outputX = pd.concat([final_outputX, dummy_city1], axis = 'columns')
final_outputX = final_outputX.drop(['city', 'ethnic_type'], axis = 'columns')


# %%
# Split training/test datasets
X_train, X_test, y_train, y_test = train_test_split(X, yDF, test_size=0.33, \
    random_state=1)
#Scale and transform
scaler = StandardScaler()
x_scaler=scaler.fit(X_train)
X_train_scaled = x_scaler.transform(X_train)
X_test_scaled = x_scaler.fit_transform(X_test)
x_test_data_scaled = x_scaler.fit_transform(final_outputX)                     



# %%
# Create a model
rf_model = RandomForestRegressor(n_estimators=50, max_depth=32, random_state=2)

# %%
import warnings;
warnings.filterwarnings('ignore');
#Fit the model using scaled trained data
rf_model = rf_model.fit(X_train_scaled, y_train)

# %%
# Verify the model with test data
y_pred = rf_model.predict(X_test_scaled)

# %%
#Score found for random forest regressor
out = rf_model.score(X_test_scaled,y_test)
f.write("Accuracy Score"%out) 

# %%
#Predictions for entire dataset with unique features
ynew = rf_model.predict(x_test_data_scaled)

# %%
#Score test data
out=rf_model.score(x_test_data_scaled,ynew)
f.write("Accuracy Score"%out) 

# %%
#Create dataframe to load into database
reviews_for_output['prediction']=np.round(ynew)

reviews_for_output['prediction']=reviews_for_output['prediction'].astype(str).\
    replace('\.0', '', regex=True)


# %%
#Insert data into postgres
try:
        reviews_for_output.to_sql(name='review_prediction', con=engine, \
        if_exists='replace',index=True)
except psycopg2.DatabaseError as error:
    f.write("Review Predition issue")   
    f.close()
    sys.exit()

# %%
#Create a .csv file
try:
    reviews_for_output.to_csv('review_prediction.csv')
except IOError as error:
    f.write("Review Predition issue")   
    f.close()
    sys.exit()

# %%
#File close
f.close()
