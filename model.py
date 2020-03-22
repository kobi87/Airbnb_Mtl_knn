import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# import the dataframe
montreal_listing =pd.read_csv('df_mtl_features_engineering.csv')
    
# Feature Scaling¶
# MinMaxScaler (proposed 1)
# we use minmaxscaler because we haven't a negative value, however we use normalised scaler (as a général standerscaler)

feature_scale=[feature for feature in montreal_listing.columns if feature not in ['host_id','price']]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(montreal_listing[feature_scale])

scaler.transform(montreal_listing[feature_scale])

# transform the data, and add on the host_id and price variables
montreal_listing = pd.concat([montreal_listing[['host_id', 'price']].reset_index(drop=True), pd.DataFrame(scaler.transform(montreal_listing[feature_scale]), columns=feature_scale)], axis=1)

# Feature selection
# Data filtering
# Filter the dataset for prices between 0 and $120
montreal_listing = montreal_listing.loc[(montreal_listing['price'] < 120)]

## for feature slection

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

#Defining the independent variables and dependent variables
airbnb_en=montreal_listing.copy()
x = airbnb_en.iloc[:,[2,3,8]]

# use log10 for the price for a good result
#y = airbnb_en['price'].values
y = airbnb_en['price']

#Getting Test and Training Set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=353)

# to visualise al the columns in the dataframe
#pd.pandas.set_option('display.max_columns', None)
#feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function#
#feature_sel_model.fit(x_train, y_train)
#feature_sel_model.get_support()

# this is how we can make a list of the selected features
#selected_feat = x_train.columns[(feature_sel_model.get_support())]
#x_train=x_train[selected_feat#
#x_test =x_test[selected_feat] 

# LR Prediction Model
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import r2_score
#from sklearn.metrics import * # importer tout les metrics d'erreurs

#Prepare a Linear Regression (knn) Model
knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')  # création une instance 
knn.fit(x_train, y_train)

# Saving model to disk
pickle.dump(knn, open('model.pkl','wb'))