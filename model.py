
# coding: utf-8

# In[ ]:
import pandas as pd
test = pd.read_csv('path..../test.csv', encoding="latin-1")

## Create function predict, take in test dataframe
def predict(test):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import preprocessing
    from datetime import datetime as dt
    from sklearn.ensemble import RandomForestRegressor
    # Read the original data for the train dataset
    df = pd.read_csv('https://raw.githubusercontent.com/ys3006/DS-Task/master/data.csv?token=AeROUwxE_C37FssnDs5buJnuqusX9Elhks5ahPaTwA%3D%3D', encoding="latin-1")

    # get only valid data df2
    df1 = df[np.isfinite(df['hammer_price'])]
    df2 = df1[(df1['hammer_price'] > 0)]
    
    # merge test to train df2
    df = pd.concat([test,df2], axis=0)
    
    # since auction_date share the same format, convert to numeric
    def auc_date(x):
        x['auction_date'] = pd.to_datetime(x['auction_date']).apply(lambda x: x.toordinal())
        
    auc_date(df)
    
    # dummy the location variable first given test and train has different size of category
    # Fill missing data with the word "Missing"
    df['location'].fillna("Missing", inplace=True)
    # Create dummy variables
    dummies = pd.get_dummies(df['location'], prefix='location')
    # Update x to include dummies and drop the original variables
    df = pd.concat([df, dummies], axis=1)
    df.drop(['location'], axis=1, inplace=True)

    #Before spliting dataset to test and train, clean the data first

    # Fill the missing data
    # Given the std of birth and death year are small and quite same, using the mean to fill NaN
    # For auction_date, estimate and measurement data, since they're highly skewed, using median instead
    def impute(x):
        x.loc[:,'artist_birth_year'].fillna(x.loc[:,'artist_birth_year'].mean(), inplace=True)
        x.loc[:,'artist_death_year'].fillna(x.loc[:,'artist_death_year'].mean(), inplace=True)

        # impute estimate price range with median
        x.loc[:,'estimate_high'].fillna(x.loc[:,'estimate_high'].median(), inplace=True)
        x.loc[:,'estimate_low'].fillna(x.loc[:,'estimate_low'].median(), inplace=True)
        x.loc[:,'measurement_depth_cm'].fillna(x.loc[:,'measurement_depth_cm'].median(), inplace=True)
        x.loc[:,'measurement_height_cm'].fillna(x.loc[:,'measurement_height_cm'].median(), inplace=True)
        x.loc[:,'measurement_width_cm'].fillna(x.loc[:,'measurement_width_cm'].median(), inplace=True)
        x.loc[:,'auction_date'].fillna(x.loc[:,'auction_date'].median(), inplace=True)

    impute(df)
    
    # Get new categorical, exclude location, which has already been dummies
    categorical_variables2 = ['artist_name',
                              'artist_nationality',
                              'category','currency']
    
    # Get dummies for the other categorical vars
    for variable in categorical_variables2:
        # Fill missing data with the word "Missing"
        df[variable].fillna("Missing", inplace=True)
            # Create dummy variables
        dummies = pd.get_dummies(df[variable], prefix=variable)
            # Update x to include dummies and drop the original variables
        df = pd.concat([df, dummies], axis=1)
        df.drop([variable], axis=1, inplace=True)
        
        
    # Given the number of unique category are too large, dropping out these four vars
    df = df.drop(['edition', 'materials', 'title', 'year_of_execution'], axis=1)
        
    
    # Split: extract test data with new created dummies location
    l = len(test)
    test2 = df.iloc[:l,]
    train = df.iloc[l:,]
    
    # get train x and train y from train dataset
    y_train = np.log(train.pop('hammer_price'))
    x_train = train
    
    # get test x and test y from test dataset
    y_test = np.log(test2.pop('hammer_price'))
    x_test = test2
    
    # Fit Random Forest Model
    RFmodel = RandomForestRegressor(1000, oob_score=True, n_jobs=-1, min_samples_leaf=1, random_state=42)
    RFmodel.fit(x_train,y_train)
    
    # Predict y by x_test with RF model
    # print output for y_train as predicted price of each row in test dataset
    y_predicted = RFmodel.predict(x_test)
    output = pd.DataFrame(y_predicted, columns = ['y_train'])
    print(output)
    
    # get RMSE for RF model 
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    
    RMSE = sqrt(mean_squared_error(y_test, y_predicted))
    print(RMSE)

    
# run function 'predict'
predict(test)
