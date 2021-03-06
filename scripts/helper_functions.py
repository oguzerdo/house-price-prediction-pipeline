import argparse
import os

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

# command line access for debuging
def get_namespace():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.add_argument('--tuning', dest='tuning', action='store_true')
    parser.add_argument('--no-tuning', dest='tuning', action='store_false')
    parser.set_defaults(debug=True)
    parser.set_defaults(tuning=True)
    return parser.parse_args()

def get_train_dataframe():
    return pd.read_csv("data/train.csv")


def get_test_dataframe():
    return pd.read_csv("data/test.csv")


def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car

def hist_for_numeric_columns(dataframe, numeric_columns):
    """
    -> Sayısal değişkenlerin histogramını çizdirir.
    :param dataframe: İşlem yapılacak dataframe.
    :param numeric_columns: Sayısal değişkenlerin adları
    """
    col_counter = 0

    data = dataframe.copy()

    for col in numeric_columns:
        data[col].hist(bins=20)

        plt.xlabel(col)

        plt.title(col)

        plt.show()

        col_counter += 1

    print(col_counter, "variables have been plotted!")

def num_summary(dataframe, col_name, plot=False):
    print(dataframe[col_name].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T)
    if plot:
        dataframe[col_name].hist(bins=10)
        script_dir = os.path.dirname(__file__)
        result_dir = script_dir + '/../outputs/num_col_graphs/'
        plt.savefig(f"{result_dir}/{col_name}.png")


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        'Ratio': dataframe[col_name].value_counts() * 100 / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        script_dir = os.path.dirname(__file__)
        result_dir = script_dir + '/../outputs/cat_col_graphs/'
        plt.savefig(f"{result_dir}/{col_name}.png")


def find_correlation(dataframe, numeric_cols, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "SalePrice":
            pass
        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


def check_dataframe(dataframe, row=5):
    print('####################  Shape   ####################')
    print(dataframe.shape)
    print('####################  Types   ####################')
    print(dataframe.dtypes)
    print('####################   Head   ####################')
    print(dataframe.head(row))
    print('####################   Tail   ####################')
    print(dataframe.tail(row))
    print('####################    NA    ####################')
    print(dataframe.isnull().sum())
    print('#################### Describe ####################')
    print(dataframe.describe().T)


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe



def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


def feature_selection(X, y):
    cols = X.columns
    # Backward Elimination
    cols = list(X.columns)
    pmax = 1
    while (len(cols) > 0):
        p = []
        X_1 = X[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y, X_1).fit()
        p = pd.Series(model.pvalues.values[1:], index=cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if (pmax > 0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols
    return selected_features_BE

def all_models(X, y, test_size=0.2, random_state=42, classification=False, holdout=False, cv_value=10, return_=True):
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.model_selection import train_test_split, cross_val_score
    # Tum Base Modeller (Classification)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    # Tum Base Modeller (Regression)
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    all_models = []

    if classification:
        models = [('CART', DecisionTreeClassifier(random_state=random_state)),
                  ('RF', RandomForestClassifier(random_state=random_state)),
                  ('GBM', GradientBoostingClassifier(random_state=random_state)),
                  ('XGBoost', XGBClassifier(random_state=random_state)),
                  ("LightGBM", LGBMClassifier(random_state=random_state)),
                  ("CatBoost", CatBoostClassifier(verbose=False, random_state=random_state))]
        if holdout:
            for name, model in models:
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                acc_train = accuracy_score(y_train, y_pred_train)
                acc_test = accuracy_score(y_test, y_pred_test)
                values = dict(name=name, acc_train=acc_train, acc_test=acc_test)
                all_models.append(values)
        else:  # For cross validation
            for name, model in models:
                model.fit(X, y)
                CV_SCORE = cross_val_score(model, X=X, y=y, cv=cv_value)
                values = dict(name=name, CV_SCORE_STD=CV_SCORE.std(), CV_SCORE_MEAN=CV_SCORE.mean())
                all_models.append(values)
        sort_method = False
    else:  # For Regression
        models = [('CART', DecisionTreeRegressor(random_state=random_state)),
                  ('RF', RandomForestRegressor(random_state=random_state)),
                  ('GBM', GradientBoostingRegressor(random_state=random_state)),
                  ("XGBoost", XGBRegressor(random_state=random_state)),
                  ("LightGBM", LGBMRegressor(random_state=random_state)),
                  ("CatBoost", CatBoostRegressor(verbose=False, random_state=random_state))]

        if holdout:
            for name, model in models:
                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)
                y_pred_train = model.predict(X_train)
                rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
                rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
                values = dict(name=name, RMSE_TRAIN=rmse_train, RMSE_TEST=rmse_test)
                all_models.append(values)
        else:  # For cross validation
            for name, model in models:
                model.fit(X, y)
                CV_SCORE = np.sqrt(-cross_val_score(estimator=model, X=X_train, y=y_train, cv=cv_value,
                                                    scoring="neg_mean_squared_error"))
                values = dict(name=name, CV_SCORE_STD=CV_SCORE.std(), CV_SCORE_MEAN=CV_SCORE.mean())
                all_models.append(values)

        sort_method = True

    if return_:
        all_models_df = pd.DataFrame(all_models)
        all_models_df = all_models_df.sort_values(all_models_df.columns[2], ascending=sort_method)
        print(all_models_df)
        return all_models_df
    else:
        all_models_df = pd.DataFrame(all_models)
        all_models_df = all_models_df.sort_values(all_models_df.columns[2], ascending=sort_method)
        print(all_models_df)

    del all_models

