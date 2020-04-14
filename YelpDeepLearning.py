#%%
#Random forest and DeepLearning
# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import tensorflow as tf
from sqlalchemy import create_engine
#%%
# The cleaned data is loaded into postgres database. It is also formatted in the format required for ML during transformation process.
# We are trying random forest classifier as the data will be divided into smaller sets and prediction could be near to accuracy
# We are also adding deep learning to get more neural network predition
# Based on the line identified, the output variable will be predicted for the input vairable
# Once the complete dataset is loaded and the accuracy is identified, we will pick the best approch. This should be sometime in next session

# %%
#Pull data from busiensses table from postgres
engine = create_engine('postgresql+psycopg2://postgres:postgres@localhost/Yelp')

# %%
ReviewsDF = pd.read_sql('select * from reviews',engine)
ReviewsDF.head()

# %%
#Identify your label and features
X = ReviewsDF.review_id.values.reshape(-1, 1)
X.shape

# %%
y = ReviewsDF.stars

# %%
# Split training/test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# %%
# Create a StandardScaler instance
scaler = StandardScaler()

# %%
# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# %%
# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# %%
# Create a random forest classifier.
rf_model = RandomForestClassifier(n_estimators=128, random_state=78)

# Fitting the model
rf_model = rf_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test_scaled)

# %%
#Get dataset ready for Database load
Output_df= pd.DataFrame(y_pred)
Output_df = Output_df.rename(columns={0:'RunPredition'})

# %%
X_df = pd.DataFrame(X_test_scaled)
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
X_df.to_sql(name='output_table', con=engine, if_exists='replace' ,index=False)
#%%

# %%
# Define the model - deep neural net
number_input_features = len(X_train_scaled[0])
hidden_nodes_layer1 =  10
hidden_nodes_layer2 = 3

nn = tf.keras.models.Sequential()
# First hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu")
)
# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))
# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# %%
# Compile the Sequential model together and customize metrics
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# %%
# Train the model
fit_model = nn.fit(X_train_scaled, y_train, epochs=50)

#%%
# make a prediction
ynew = nn.predict_classes(X_test_scaled)

# %%
#Get dataset ready for Database load
Output_deep_df= pd.DataFrame(ynew)
Output_deep_df = Output_deep_df.rename(columns={0:'RunPredition'})

# %%
XScaled_df = pd.DataFrame(X_test_scaled)
XScaled_df = XScaled_df.rename(columns={0:'Input'})
XScaled_df.head()


# %%
XScaled_df['RunPredition']=Output_deep_df['RunPredition']
XScaled_df.head()

#%%
#Import data into postgres
XScaled_df.to_sql(name='output_deep', con=engine, if_exists='replace' ,index=False)
# %%
# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# %%
