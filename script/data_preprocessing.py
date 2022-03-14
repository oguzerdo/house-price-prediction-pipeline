from script.helper_functions import *


train = get_train_dataframe()
test = get_test_dataframe()
dataframe = train.append(test).reset_index(drop=True)

def data_preprocessing(dataframe):
# Buradaki eksik gözlemler o özelliğin olmadığı anlamına gelmekte. Bu yüzden None atayacağım.

    none_cols = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageType',
                 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
                 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']

    # Sayısal değişkenlerdeki eksik gözlemler de aynı şekilde, bunlara 0 atıyorum.

    zero_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath','BsmtHalfBath', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea']

    # Bu değişkenlerdeki eksik gözlemlere mod atayacağım.

    freq_cols = ['Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual','SaleType', 'Utilities']


    for col in zero_cols:
        dataframe[col].replace(np.nan, 0, inplace=True)

    for col in none_cols:
        dataframe[col].replace(np.nan, 'None', inplace=True)

    for col in freq_cols:
        dataframe[col].replace(np.nan, dataframe[col].mode()[0], inplace=True)


    # MsZoning değişkenindeki boş değerleri MSSubClassa göre doldurma.

    dataframe['MSZoning'] = dataframe.groupby('MSSubClass')['MSZoning'].apply(lambda x: x.fillna(x.mode()[0]))

    # LotFrontage mülkiyetin cadde ile bağlantısını gösteren bir değişken, her mahallenin cadde bağlantısının birbirine benzeyebileceğinden bunu Neighborhood'a a göre doldurdum.

    dataframe['LotFrontage'] = dataframe.groupby(['Neighborhood'])['LotFrontage'].apply(lambda x: x.fillna(x.median()))

    # Sayısal değişken olup aslında kategorik değişken olması gerekenleri düzeltme

    dataframe['MSSubClass'] = dataframe['MSSubClass'].astype(str)
    dataframe['YrSold'] = dataframe['YrSold'].astype(str)
    dataframe['MoSold'] = dataframe['MoSold'].astype(str)

    # Neighboor içerisindeki benzer değerde olanları birbiri ile grupladım.

    neigh_map = {'MeadowV': 1,'IDOTRR': 1,'BrDale': 1,'BrkSide': 2,'OldTown': 2,'Edwards': 2,
                 'Sawyer': 3,'Blueste': 3,'SWISU': 3,'NPkVill': 3,'NAmes': 3,'Mitchel': 4,
                 'SawyerW': 5,'NWAmes': 5,'Gilbert': 5,'Blmngtn': 5,'CollgCr': 5,
                 'ClearCr': 6,'Crawfor': 6,'Veenker': 7,'Somerst': 7,'Timber': 8,
                 'StoneBr': 9,'NridgHt': 10,'NoRidge': 10}

    dataframe['Neighborhood'] = dataframe['Neighborhood'].map(neigh_map).astype('int')


    # Derecelendirme içeren değişkenleri ordinal yapıya getirdim.

    ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    dataframe['ExterQual'] = dataframe['ExterQual'].map(ext_map).astype('int')

    ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    dataframe['ExterCond'] = dataframe['ExterCond'].map(ext_map).astype('int')

    bsm_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    dataframe['BsmtQual'] = dataframe['BsmtQual'].map(bsm_map).astype('int')
    dataframe['BsmtCond'] = dataframe['BsmtCond'].map(bsm_map).astype('int')

    bsmf_map = {'None': 0,'Unf': 1,'LwQ': 2,'Rec': 3,'BLQ': 4,'ALQ': 5,'GLQ': 6}
    dataframe['BsmtFinType1'] = dataframe['BsmtFinType1'].map(bsmf_map).astype('int')
    dataframe['BsmtFinType2'] = dataframe['BsmtFinType2'].map(bsmf_map).astype('int')

    heat_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    dataframe['HeatingQC'] = dataframe['HeatingQC'].map(heat_map).astype('int')
    dataframe['KitchenQual'] = dataframe['KitchenQual'].map(heat_map).astype('int')
    dataframe['FireplaceQu'] = dataframe['FireplaceQu'].map(bsm_map).astype('int')
    dataframe['GarageCond'] = dataframe['GarageCond'].map(bsm_map).astype('int')
    dataframe['GarageQual'] = dataframe['GarageQual'].map(bsm_map).astype('int')


    # Dropping outliers after detecting them by eye.
    dataframe.loc[2590, 'GarageYrBlt'] = 2007 # missing value it was 2207

    dataframe = dataframe.drop(dataframe[(dataframe['OverallQual'] < 5) & (dataframe['SalePrice'] > 200000)].index)
    dataframe = dataframe.drop(dataframe[(dataframe['GrLivArea'] > 4000) & (dataframe['SalePrice'] < 200000)].index)
    dataframe = dataframe.drop(dataframe[(dataframe['GarageArea'] > 1200) & (dataframe['SalePrice'] < 200000)].index)
    dataframe = dataframe.drop(dataframe[(dataframe['TotalBsmtSF'] > 3000) & (dataframe['SalePrice'] > 320000)].index)
    dataframe = dataframe.drop(dataframe[(dataframe['1stFlrSF'] < 3000) & (dataframe['SalePrice'] > 600000)].index)
    dataframe = dataframe.drop(dataframe[(dataframe['1stFlrSF'] > 3000) & (dataframe['SalePrice'] < 200000)].index)


    ##################
    # Outliers
    ##################

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    num_cols = [col for col in num_cols if col not in ["SALEPRICE"]]

    # for col in num_cols:
    #    print(col, check_outlier(dataframe, col, q1=0.01, q3=0.99))

    for col in num_cols:
        replace_with_thresholds(dataframe, col, q1=0.01, q3=0.99)

    #dataframe.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T

    return dataframe

