
#%%
import requests
import json
 

# %%
api_key = "XhNX92521J_CznXDhlJx8-fGw93UzjmrVnmC41ZVFfDIc0ePgHoMB472mNVNize3LCT7oPbr1ttiulBtGWhIi83SMORqaJaZDjDytarEPUA4bLQgDLj029H44_CPXnYx"
headers = {'Authorization': 'Bearer %s' % api_key}
 

# %%

url='https://api.yelp.com/v3/businesses/search'

# In the dictionary, term can take values like food, cafes or businesses like McDonalds
params = {'term':'seafood','location':'New York City'}


# %%
req=requests.get(url, params=params, headers=headers)
 
# proceed only if the status code is 200
print('The status code is {}'.format(req.status_code))

# %%

# printing the text from the response 
json.loads(req.text)

#%%

url = "https://api.yelp.com/v3/businesses/FEVQpbOPOwAPNIgO7D3xxw/reviews"
req = requests.get(url, headers=headers)
print('the status code is {}'.format(req.status_code))

#%%
json.loads(req.text)

# %%
