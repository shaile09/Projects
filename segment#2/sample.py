#%%
import pandas as pd
import seaborn as sns
import requests, re
import pandas as pd
from sklearn.cluster import KMeans
import json
from pandas.io.json import json_normalize

# %% [markdown]
# ### Clean Yelp_business dataset 

# %%
with open(f'yelp_dataset/dataset/business.json', mode='r', encoding="utf8") as file:
    yelp_source = [json.loads(line) for line in file]


#%%
business = pd.DataFrame(yelp_source)
business.head()

# %%
## drop unuseful column 'neighborhood' 
business.drop(['neighborhood'], axis=1, inplace=True)

## remove quotation marks in name and address column
business.name=business.name.str.replace('"','')
business.address=business.address.str.replace('"','')

## filter restaurants of US
# states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
#           "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
#           "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
#           "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
#           "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

states = ["AZ"]
usa=business.loc[business['state'].isin(states)]
usa.head()

#%%
#usa = usa.head(n=200)

# %%
## select all restaurants in USA
usa['category'] = usa['categories'].apply(lambda x: ','.join(map(str, x)))
usa.head()

#%%
us_restaurants=usa[usa['category'].str.contains('Restaurants')]

#%%

#%%
## select out 16 cuisine types of restaurants and rename the category
us_restaurants.is_copy=False
#us_restaurants['category']=pd.Series()
us_restaurants.loc[us_restaurants.category.str.contains('American'),'category'] = 'American'
us_restaurants.loc[us_restaurants.category.str.contains('Mexican'), 'category'] = 'Mexican'
us_restaurants.loc[us_restaurants.category.str.contains('Italian'), 'category'] = 'Italian'
us_restaurants.loc[us_restaurants.category.str.contains('Japanese'), 'category'] = 'Japanese'
us_restaurants.loc[us_restaurants.category.str.contains('Chinese'), 'category'] = 'Chinese'
us_restaurants.loc[us_restaurants.category.str.contains('Thai'), 'category'] = 'Thai'
us_restaurants.loc[us_restaurants.category.str.contains('Mediterranean'), 'category'] = 'Mediterranean'
us_restaurants.loc[us_restaurants.category.str.contains('French'), 'category'] = 'French'
us_restaurants.loc[us_restaurants.category.str.contains('Vietnamese'), 'category'] = 'Vietnamese'
us_restaurants.loc[us_restaurants.category.str.contains('Greek'),'category'] = 'Greek'
us_restaurants.loc[us_restaurants.category.str.contains('Indian'),'category'] = 'Indian'
us_restaurants.loc[us_restaurants.category.str.contains('Korean'),'category'] = 'Korean'
us_restaurants.loc[us_restaurants.category.str.contains('Hawaiian'),'category'] = 'Hawaiian'
us_restaurants.loc[us_restaurants.category.str.contains('African'),'category'] = 'African'
us_restaurants.loc[us_restaurants.category.str.contains('Spanish'),'category'] = 'Spanish'
us_restaurants.loc[us_restaurants.category.str.contains('Middle_eastern'),'category'] = 'Middle_eastern'
us_restaurants.loc[us_restaurants.category.str.contains('Fast Food'),'category'] = 'American'
us_restaurants.loc[us_restaurants.category.str.contains('Burgers'),'category'] = 'American'
us_restaurants.loc[us_restaurants.category.str.contains('Pizza'),'category'] = 'American'
us_restaurants.loc[us_restaurants.category.str.contains('Seafood'),'category'] = 'SeaFood'
us_restaurants.loc[us_restaurants.category.str.contains('Food Stands'),'category'] = 'Food Stands'
us_restaurants.loc[us_restaurants.category.str.contains('Barbeque'),'category'] = 'American'

#us_restaurants.category[:20]

#%%

#%%
us_restaurants.head()

#%%
len(us_restaurants['city'].unique())

#%%
us_restaurants['city'].value_counts()

#%%
city = ['Phoenix']
us_restaurants = us_restaurants[us_restaurants['city'].isin(city)]
us_restaurants.head(n=20)

#%%
len(us_restaurants['postal_code'].unique())

#%%
print(us_restaurants['postal_code'].value_counts())

# %%
# label reviews as positive or negative or neural
us_restaurants['labels'] = ''
us_restaurants.loc[us_restaurants.stars >=4, 'labels'] = 'positive'
us_restaurants.loc[us_restaurants.stars ==3, 'labels'] = 'neural'
us_restaurants.loc[us_restaurants.stars <3, 'labels'] = 'negative'
us_restaurants.head()

# %%
us_restaurants.drop(['attributes', 'hours'], axis=1, inplace=True)
us_restaurants.head()

# %%
us_restaurants.drop(['categories', 'is_open', 'address'], axis=1, inplace=True)
us_restaurants.head()

# %%
postalcodes = us_restaurants.groupby(['postal_code'])['postal_code','category','stars','review_count']

# %%
postalcodes.head()


# %%
postalcodes['85022']

# %%
