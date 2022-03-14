import pickle as pickle
from script.data_preprocessing import *

train = get_train_dataframe()
test = get_test_dataframe()
df = train.append(test).reset_index(drop=True)

dataframe = data_preprocessing(df)

def feature_engineering(dataframe):
    # Creating new dataframe  based on previous observations. There might be some highly correlated dataframe now. You cab drop them if you want to...

    dataframe['TotalSF'] = (dataframe['BsmtFinSF1'] + dataframe['BsmtFinSF2'] + dataframe['1stFlrSF'] + dataframe['2ndFlrSF'])
    dataframe['TotalBathrooms'] = (dataframe['FullBath'] + (0.5 * dataframe['HalfBath']) + dataframe['BsmtFullBath'] + (0.5 * dataframe['BsmtHalfBath']))

    dataframe['TotalPorchSF'] = (dataframe['OpenPorchSF'] + dataframe['3SsnPorch'] + dataframe['EnclosedPorch'] + dataframe['ScreenPorch'] + dataframe['WoodDeckSF'])

    dataframe['YearBlRm'] = (dataframe['YearBuilt'] + dataframe['YearRemodAdd'])

    # Merging quality and conditions.

    dataframe['TotalExtQual'] = (dataframe['ExterQual'] + dataframe['ExterCond'])
    dataframe['TotalBsmQual'] = (dataframe['BsmtQual'] + dataframe['BsmtCond'] + dataframe['BsmtFinType1'] + dataframe['BsmtFinType2'])
    dataframe['TotalGrgQual'] = (dataframe['GarageQual'] + dataframe['GarageCond'])
    dataframe['TotalQual'] = dataframe['OverallQual'] + dataframe['TotalExtQual'] + dataframe['TotalBsmQual'] + dataframe[ 'TotalGrgQual'] + dataframe['KitchenQual'] + dataframe['HeatingQC']

    # Creating new dataframe by using new quality indicators.

    dataframe['QualGr'] = dataframe['TotalQual'] * dataframe['GrLivArea']
    dataframe['QualBsm'] = dataframe['TotalBsmQual'] * (dataframe['BsmtFinSF1'] + dataframe['BsmtFinSF2'])
    dataframe['QualPorch'] = dataframe['TotalExtQual'] * dataframe['TotalPorchSF']
    dataframe['QualExt'] = dataframe['TotalExtQual'] * dataframe['MasVnrArea']
    dataframe['QualGrg'] = dataframe['TotalGrgQual'] * dataframe['GarageArea']
    dataframe['QlLivArea'] = (dataframe['GrLivArea'] - dataframe['LowQualFinSF']) * (dataframe['TotalQual'])
    dataframe['QualSFNg'] = dataframe['QualGr'] * dataframe['Neighborhood']

    dataframe["new_home"] = dataframe["YearBuilt"]
    dataframe.loc[dataframe["new_home"] == dataframe["YearRemodAdd"], "new_home"] = 0
    dataframe.loc[dataframe["new_home"] != dataframe["YearRemodAdd"], "new_home"] = 1

    # Creating some simple dataframe.

    dataframe['HasPool'] = dataframe['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    dataframe['Has2ndataframeloor'] = dataframe['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    dataframe['HasGarage'] = dataframe['QualGrg'].apply(lambda x: 1 if x > 0 else 0)
    dataframe['HasBsmt'] = dataframe['QualBsm'].apply(lambda x: 1 if x > 0 else 0)
    dataframe['HasFireplace'] = dataframe['Fireplaces'].apply(lambda x: 1if x > 0 else 0)
    dataframe['HasPorch'] = dataframe['QualPorch'].apply(lambda x: 1 if x > 0 else 0)

    ###### Rare Encoding ######
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    dataframe = rare_encoder(dataframe, 0.01)

    useless_cols = [col for col in cat_cols if dataframe[col].nunique() == 1 or
                    (dataframe[col].nunique() == 2 and (dataframe[col].value_counts() / len(dataframe) <= 0.01).any(
                        axis=None))]

    cat_cols = [col for col in cat_cols if col not in useless_cols]

    for col in useless_cols:
        dataframe.drop(col, axis=1, inplace=True)

    ###### Label Encoding ######
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in [np.int64, np.float64]
                   and dataframe[col].nunique() == 2]

    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    ###### One-Hot Encoding ######
    cat_cols = cat_cols + cat_but_car

    dataframe = one_hot_encoder(dataframe, cat_cols, drop_first=False)

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    useless_cols_new = [col for col in cat_cols if
                        (dataframe[col].value_counts() / len(dataframe) <= 0.01).any(axis=None)]

    for col in useless_cols_new:
        dataframe.drop(col, axis=1, inplace=True)

    ###### Saving Test and Train dataframe as pickle ######
    train_dataframe = dataframe[dataframe["SalePrice"].notnull()]
    test_dataframe = dataframe[dataframe["SalePrice"].isnull()].drop("SalePrice", axis=1)

    curr_dir = os.getcwd()
    os.makedirs("outputs/pickles", exist_ok=True)
    result_dir = curr_dir + '/outputs/pickles/'

    with open(result_dir + 'train_dataframe.pkl', 'wb') as f:
        pickle.dump(train_dataframe, f)

    with open(result_dir + 'test_dataframe.pkl', 'wb') as f:
        pickle.dump(test_dataframe, f)


