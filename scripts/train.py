"""
Train script
"""

import pandas as pd
import os
import json
import pickle
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from scripts.config import *
from scripts.helper_functions import *


def train_model(debug=False, tuning=True):
    cur_dir = os.getcwd()
    pickle_dir = cur_dir + '/outputs/pickles/'
    train_df = pd.read_pickle(pickle_dir + 'train_dataframe.pkl')
    print("Train dataset loaded. Observation number: ", train_df.shape[0], "\n")

    if debug:
        train_df = train_df.sample(100)
        print("Debug mode is active. Running with subsample train set...", "\n")
    else:
        pass

    y = train_df['SalePrice']
    X = train_df.drop(["SalePrice", "Id"], axis=1)
    selected_features = feature_selection(X, y)  # feature selection
    X = X[selected_features]

    # Model Report
    print("Model Report is started.")

    models = all_models(X, y, classification=False, cv_value=5, return_=True)
    best_models = models[0:3].merge(models_df, how='left', on='name')[["name", "model", "params"]]

    if tuning:
        # Automated Hyperparameter Optimization
        print("\n########### Hyperparameter Optimization ###########\n")
        for index, row in best_models.iterrows():
            name = row['name']
            model = row['model']
            params = row['params']

            print(f"########## {name} ##########")
            rmse = np.mean(
                np.sqrt(-cross_val_score(model, X, y, cv=10, scoring="neg_mean_squared_error")))  # base model rmse
            print(f"RMSE: {round(rmse, 4)} ({name}) ")

            gs_best = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)  # finding best params

            final_model = model.set_params(**gs_best.best_params_)  # save best params model
            rmse_new = np.mean(np.sqrt(
                -cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))  # tuned model rmse score
            print(f"RMSE (After): {round(rmse_new, 4)} ({name}) ")

            print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

            # MODEL SAVING PART
            today = pd.to_datetime("today").strftime('%d-%m-%Y-%H.%M')
            model_info = dict(date=today, name=name, rmse=rmse,
                              rmse_new=rmse_new, count=X.shape[0],
                              best_params=gs_best.best_params_)

            try:  # Save model info to JSON
                with open('outputs/model_info_data.json', 'r+') as f:
                    model_info_data = json.load(f)

                model_info_data['data'].append(model_info)

                with(open('outputs/model_info_data.json', 'w')) as f:
                    json.dump(model_info_data, f)
            except:  # If not JSON file
                print("No JSON file, JSON File is creating")
                with(open('outputs/model_info_data.json', 'w')) as f:
                    json.dump({'data': [model_info]}, f)

            # Save Models
            os.makedirs("outputs/pickles/models", exist_ok=True)
            model_dir = "outputs/pickles/models/"
            with open(model_dir + f'{today}-{name}-{int(rmse_new)}.pkl', 'wb') as f:
                pickle.dump(final_model, f)

        voting_model = VotingRegressor(estimators=[(best_models.iloc[0]["name"], best_models.iloc[0]["model"]),
                                                   (best_models.iloc[1]["name"], best_models.iloc[1]["model"]),
                                                   (best_models.iloc[2]["name"], best_models.iloc[2]["model"])])

        voting_rmse = np.mean(np.sqrt(-cross_val_score(voting_model, X, y, cv=10, scoring="neg_mean_squared_error")))

        print("\n########## Best Models ##########\n")
        print(pd.json_normalize(json.load(open("outputs/model_info_data.json", 'r'))["data"], max_level=0).sort_values(
            'date', ascending=False)[0:3][["name", "rmse_new"]])
        print("\n########## Voting Regressor ##########\n")
        print(f"RMSE: {round(voting_rmse, 4)} (Voting Regressor) ")
        voting_model.fit(X, y)

        # Save voting_model
        with open(model_dir + f'{today}-VotingModel-{int(voting_rmse)}.pkl', 'wb') as f:
            pickle.dump(voting_model, f)

        return voting_model

    else:
        pass

