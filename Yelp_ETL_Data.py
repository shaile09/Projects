#%%
# Import Dependancies
import requests
import json
from pandas.io.json import json_normalize
import pandas as pd

#%%
# Import business JSON file 
file_dir = 'C:/Users/knush/GitRepository/Class_Practice/FinalProject/yelp-dataset/'
f'{file_dir}yelp_academic_dataset_business.json'


#%%
# Open business JSON file
with open(f'{file_dir}yelp_academic_dataset_business.json', mode='r', encoding="utf8", errors='ignore') as f:
    buss_data = [json.loads(line) for line in f]

#%%
# Import to dataframe
bussiness_df = pd.DataFrame(buss_data)
bussiness_df.head()

#%%
bussiness_df.count()
#%%
CopyDF = bussiness_df.copy()

#%%
CopyDF.head()

# %%
# Remove all closed restaurants
a = [0]
CopyDF = CopyDF[~CopyDF['is_open'].isin(a)]

# %%
# If we need to drop any columns
#CopyDF = CopyDF.drop(columns=[])

# %%
len(CopyDF['stars'].unique())

#%%
CopyDF['stars'].value_counts()

#%%
#CopyDF['stars_id'] = CopyDF['stars'].apply(lambda x : x if x in ('4.0','4.5','5.0') else 'good')

# %%
CopyDF.head()
CopyDF.count()

#%%
# Filter out all the data that are restaraunts


# %%
#Import Business review JSON file
file_dir = 'C:/Users/knush/GitRepository/Class_Practice/FinalProject/yelp-dataset/'
f'{file_dir}yelp_academic_dataset_review.json'


#%%
# Open business JSON file
with open(f'{file_dir}yelp_academic_dataset_review.json', mode='r', encoding="utf8", errors='ignore') as f:
    reviews_data = [json.loads(line) for line in f]

#%%
# Import to dataframe
reviews_df = pd.DataFrame(reviews_data)
reviews_df.head()

#%%
reviews_df.count()

#%%
# Modify this code to only pull reviews for business dataframe

# dfAllReviews = pd.DataFrame(columns=['id', 'url', 'text','rating','time_created','user.id','user.profile_url','user.image_url','user.name'])
# for ind in df.index:     
#     URL_REVIEWS = url + df['id'][ind]+'/reviews'
#     reqReview=requests.get(URL_REVIEWS, headers=headers)
#     json.loads(reqReview.text)
#     dataReview = json.loads(reqReview.text)   
#     if not (dataReview['reviews']):
#         print('if')
#         print(ind)        
#         print('noreviews')
#     else:   
#         print('else')
#         print(ind)        
#         dfReview = pd.DataFrame.from_dict(json_normalize(dataReview['reviews']), orient='columns')
#         dfAllReviews=dfAllReviews.append(dfReview,ignore_index=True)
# %%
#dfAllReviews.head()

# %%
# Clean the data for machine learning model, analysis and visuliazation

#%%
# After cleaning the data put in dataframe to match DataBase table structure

#%%
# Upload the data to the posgres database