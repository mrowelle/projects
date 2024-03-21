import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import catboost
import xgboost
from sklearn.model_selection import train_test_split
from catboost.datasets import titanic
import math
pd.set_option('display.max_rows',None)

train_df, test_df = titanic()

train_df['Cabin'] = train_df['Cabin'].replace(np.NaN, 'U')
train_df['Cabin'] = train_df['Cabin'].apply(lambda x: x[:1])
train_df['Cabin'] = train_df['Cabin'].replace('U', 'Unknown')
train_df['Cabin'].head()
#titanic_df['Cabin'] = titanic_df['Cabin'].str.slice(start = 1)

test_df['Cabin'] = test_df['Cabin'].replace(np.NaN, 'U')
test_df['Cabin'] = test_df['Cabin'].apply(lambda x: x[:1])
test_df['Cabin'] = test_df['Cabin'].replace('U', 'Unknown')
test_df['Cabin'].head()

train_df['isFemale'] = np.where(train_df['Sex'] == 'female', 1,0)
train_df['Pclass'] = np.where(train_df['Pclass']==1, 'First', 
                                np.where(train_df['Pclass']==2, 'Second','Third'))

train_df.isna().sum()
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())
train_df = train_df.dropna()

test_df.isna().sum()

test_df['isFemale'] = np.where(test_df['Sex'] == 'female', 1,0)
test_df['Pclass'] = np.where(test_df['Pclass']==1, 'First', 
                                np.where(test_df['Pclass']==2, 'Second','Third'))

test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean())

train_df = train_df[['Survived','Pclass','isFemale','Age','SibSp','Parch','Cabin','Embarked']]

test_df = test_df[['Pclass','isFemale','Age','SibSp','Parch','Cabin','Embarked']]

X = train_df.drop(columns='Survived', axis=1)
y = train_df['Survived']

titanic_categories = np.where(X.dtypes != float)[0]

from catboost import CatBoostClassifier, Pool, metrics, cv
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

SEED = 1234
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=SEED)

#X_test = test_df

params = {'iterations':5000,
        'learning_rate':0.01,
        'cat_features':titanic_categories,
        'depth':3,
        'eval_metric':'AUC',
        'verbose':200,
        'od_type':"Iter", # overfit detector
        'od_wait':500, # most recent best iteration to wait before stopping
        'random_seed': SEED
          }

cat_model = CatBoostClassifier(**params)
cat_model.fit(X_train, y_train,   
          eval_set=(X_test, y_test), 
          use_best_model=False, # True if we don't want to save trees created after iteration with the best validation score
          plot=True  
         );













