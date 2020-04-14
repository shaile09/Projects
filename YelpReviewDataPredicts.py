#%%
#Import libraries
import pandas as pd
import psycopg2
import sqlalchemy as sa
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

# %%
#Pull data from busiensses table from postgres
engine = create_engine('postgresql+psycopg2://postgres:postgres@localhost/Yelp')

# %%
ReviewsDF = pd.read_sql('select * from reviews',engine)
ReviewsDF.head()
# %%
# Generate our categorical variable list
reviews_cat = ReviewsDF.dtypes[ReviewsDF.dtypes == "object"].index.tolist()
ReviewsDF[reviews_cat].nunique()
# %%

# %%
#Identify your label and features
X = ReviewsDF.bussiness_id.values.reshape(-1, 1)
X.shape

# %%
y = ReviewsDF.stars

# %%
# Create a model with scikit-learn
model = LinearRegression()

# %%
# Fit the data into the model
model.fit(X, y)

# %%
# The model creates predicted y values based on X values
y_pred = model.predict(X)

# %%
#Get dataset ready for Database load
Output_df= pd.DataFrame(y_pred)
Output_df = Output_df.rename(columns={0:'RunPredition'})

# %%
X_df = pd.DataFrame(X)
X_df = X_df.rename(columns={0:'Input'})
X_df.head()


# %%
X_df['RunPredition']=Output_df['RunPredition']
X_df.head()

# %%
from sqlalchemy.orm import Session
session = Session(engine)
#%%
#Import data into postgres
X_df.to_sql(name='output', con=engine, if_exists='replace' ,index=False)
#%%
print('Intercept: \n', model.intercept_)
print('Coefficients: \n', model.coef_)

#%%
model.score(X,y)