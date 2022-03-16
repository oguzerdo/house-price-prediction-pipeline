from scripts.train import *
import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

###### CART ######
cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

###### Random Forests ######
rf_params = {"max_depth": [20, 22, 24],
             "max_features": [30, 32, 34],
             "n_estimators": [400, 600, 800],
             "min_samples_split": [2, 4]}

###### GBM Model ######
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 4, 5],
              "n_estimators": [1600, 1800, 2000],
              "subsample": [0.2, 0.3, 0.4],
              "loss": ['huber'],
              "max_features": ['sqrt']}

###### XGBoost ######
xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 15, 20],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}

###### LightGBM ######
lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [1300, 1500, 1700],
                   "colsample_bytree": [0.2, 0.3, 0.4]}

###### CatBoost ######
catboost_params = {"iterations": [400, 500, 600],
                   "learning_rate": [0.01, 0.1],
                   "depth": [4, 5, 6, 7]}

models_ = [('CART', DecisionTreeRegressor(), cart_params),
          ('RF', RandomForestRegressor(), rf_params),
          ('GBM', GradientBoostingRegressor(), gbm_params),
          ("XGBoost", XGBRegressor(objective='reg:squarederror'), xgboost_params),
          ("LightGBM", LGBMRegressor(), lightgbm_params),
          ("CatBoost", CatBoostRegressor(verbose=False), catboost_params)]

columns = ['name', 'model', 'params']

models_df = pd.DataFrame(models_, columns=columns)

