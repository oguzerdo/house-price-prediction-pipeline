from script.train import *
import pandas as pd

today = pd.to_datetime("today").strftime('%d-%m-%Y-%H:%M')

###### CART ######
cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

###### Random Forests ######
rf_params = {"max_depth": [20, 22],
             "max_features": [30, 32]}

###### GBM Model ######
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 4, 5]}

###### XGBoost ######
xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 12]}

###### LightGBM ######
lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [1300, 1500, 1700],
                   "colsample_bytree": [0.2, 0.3, 0.4]}

###### CatBoost ######
catboost_params = {"iterations": [400, 500],
                   "learning_rate": [0.01, 0.1]}

models_ = [('CART', DecisionTreeRegressor(), cart_params),
          ('RF', RandomForestRegressor(), rf_params),
          ('GBM', GradientBoostingRegressor(), gbm_params),
          ("XGBoost", XGBRegressor(objective='reg:squarederror'), xgboost_params),
          ("LightGBM", LGBMRegressor(), lightgbm_params),
          ("CatBoost", CatBoostRegressor(verbose=False), catboost_params)]

columns = ['name', 'model', 'params']

models_df = pd.DataFrame(models_, columns=columns)

# import json
#
#
# with(open('catboost_params.json', 'w')) as f:
#     json.dump(catboost_params, f)
#
#
# with open('catboost_params.json') as f:
#     catboost_params = json.load(f)
#     print(catboost_params)
