# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
import matplotlib as plt
from sklearn.datasets import make_blobs
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Import our input dataset
business_df = pd.read_csv('yelp-dataset/fakedataset.csv')
business_df.head()


# %%
# What variable(s) are considered the target for model?
    # Y = stars_id

# What variable(s) are considered to be the features for your model?
    # X = other than stars

# What variable(s) are neither and should be removed from the input data?
    # business_id, name, categoties1, state, address, city


# %%
# drop columns not needed for the model
business_df1 = business_df.drop(["business_id", "name", "categories1", "state", "address", "city"],1)
business_df1.head()

#%%
# Drop null rows
business_df1 = business_df1.dropna()
business_df1.head()

#%%
business_df1['stars_id'] = business_df1['stars'].apply(lambda x : x if x in (4.0, 4.5, 5.0) else 0)
business_df1

#%%
business_df1['stars_id'] = business_df1['stars_id'].apply(lambda x : x if x == 0 else 1.0)
business_df1

# %%
# Generate our categorical variable list
business_cat = business_df1.dtypes[business_df1.dtypes == "object"].index.tolist()
business_cat


# %%
# Check the number of unique values in each column
business_df1[business_cat].nunique()


# %%
# Check the unique value counts to see if binning is required - 1st value
class_counts = business_df1.category2.value_counts()
print(class_counts)


# %%
# Visualize the value counts - 1st value 
class_counts.plot.density()


# %%
# Determine which values to replace - 1st value
replace_class = list(class_counts[class_counts < 100].index)

# Replace in DataFrame
for category in replace_class:
    business_df1.category2 = business_df1.category2.replace(category,"Other")
# Check to make sure binning was successful
business_df1.category2.value_counts()


# %%
# Check the unique value counts to see if binning is required - 2nd Value
# Do the same code as the 1st value for the 2nd value if have the second value 

# Determine which values to replace - 2nd value

# Replace in DataFrame

# Check to make sure binning was successful



# %%
# Check the number of unique values in each column
business_df1[business_cat].nunique()


# %%
# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Fit and transform the OneHotEncoder using the categorical variable list
encode_df = pd.DataFrame(enc.fit_transform(business_df1[business_cat]))

# Add the encoded variable names to the DataFrame
encode_df.columns = enc.get_feature_names(business_cat)
encode_df.head()


# %%
# Merge one-hot encoded features and drop the originals
business_df1 = business_df1.merge(encode_df,left_index=True, right_index=True)
business_df1 = business_df1.drop(business_cat,1)
business_df1


# %%
# Split our preprocessed data into our features and target arrays
y = business_df1["stars_id"].values
X = business_df1.drop(["stars_id"],1).values

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# %%
# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# %%
# DEEP LEARNING MODEL


# %%
len(X_train[0])


# %%
# Define the model - deep neural net
number_input_features = len(X_train[0])
hidden_nodes_layer1 =  8
hidden_nodes_layer2 = 6


nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu")
)

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="sigmoid"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()


# %%
# Compile the model
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# %%
# Train the model
fit_model = nn.fit(X_train,y_train,epochs=500)


# %%
# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


# %%
# NEURAL NETWORK MODEL


# %%
# Create the Keras Sequential model
nn_model = tf.keras.models.Sequential()

# Add our first Dense layer, including the input layer
nn_model.add(tf.keras.layers.Dense(units=1, activation="relu", input_dim=16))

# Add the output layer that uses a probability activation function
nn_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the Sequential model
nn_model.summary()


# %%
#Compile the Sequential model together and customize metrics
nn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# %%
# Fit the model to the training data
fit_model = nn_model.fit(X_train_scaled, y_train, epochs=100)


# %%
# Create a DataFrame containing training history
history_df = pd.DataFrame(fit_model.history, index=range(1,len(fit_model.history["loss"])+1))

# Plot the loss
history_df.plot(y="loss")


# %%
# Plot the accuracy
history_df.plot(y="accuracy")


# %%
# Evaluate the model using the test data
model_loss, model_accuracy = nn_model.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


# %%
# BASIC NEURAL NETWORK MODEL


# %%
# Define the basic neural network model
nn_model = tf.keras.models.Sequential()
nn_model.add(tf.keras.layers.Dense(units=16, activation="relu", input_dim=15))
nn_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Compile the Sequential model together and customize metrics
nn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
fit_model = nn_model.fit(X_train_scaled, y_train, epochs=100)

# Evaluate the model using the test data
model_loss, model_accuracy = nn_model.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


# %%
# Evaluate the model using the test data
model_loss, model_accuracy = nn_model.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


# %%
# Create a random forest classifier.
rf_model = RandomForestClassifier(n_estimators=128, random_state=78)

# Fitting the model
rf_model = rf_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test_scaled)
print(f" Random forest predictive accuracy: {accuracy_score(y_test,y_pred):.3f}")


# %%
# Define the logistic regression model
log_classifier = LogisticRegression(solver="lbfgs",max_iter=200)

# Train the model
log_classifier.fit(X_train,y_train)

# Evaluate the model
y_pred = log_classifier.predict(X_test)
print(f" Logistic regression model accuracy: {accuracy_score(y_test,y_pred):.3f}")

# %%

# Linear LinearRegression

# %%
plt.scatter(business_df1.stars, business_df1.review_count)
plt.xlabel('review_count')
plt.ylabel('stars')
plt.show()


# %%
# The data in the df column must be reshaped into an array with shape (num_samples, num_features)
# https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape
X = business_df1.stars.values.reshape(-1, 1)


# %%
X[:5]


# %%
# The shape of X is 30 samples, with a single feature (column)
X.shape


# %%
y = business_df1.review_count


# %%
# Create a model with scikit-learn
model = LinearRegression()


# %%
# Fit the data into the model
# By convention, X is capitalized and y is lowercase
model.fit(X, y)


# %%
# The model creates predicted y values based on X values
y_pred = model.predict(X)
print(y_pred.shape)


# %%
# Plot the results. The best fit line is red.
plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.show()


# %%
# The slope
# The y-intercept
print(model.coef_)
print(model.intercept_)

# %%
