# Regresyon Modelleri ile Ev Fiyat Tahmin Modeli


![banner](https://storage.googleapis.com/kaggle-competitions/kaggle/5407/media/housesbanner.png)

--

# Features

- SalePrice - mülkün dolar cinsinden satış fiyatı. Bu, tahmin etmeye çalışılan hedef değişkendir.
- MSSubClass: İnşaat sınıfı
- MSZoning: Genel imar sınıflandırması
- LotFrontage: Mülkiyetin cadde ile doğrudan bağlantısının olup olmaması
- LotArea: Parsel büyüklüğü
- Street: Yol erişiminin tipi
- Alley: Sokak girişi tipi
- LotShape: Mülkün genel şekli
- LandContour: Mülkün düzlüğü
- Utulities: Mevcut hizmetlerin türü
- LotConfig: Parsel yapılandırması
- LandSlope: Mülkün eğimi
- Neighborhood: Ames şehir sınırları içindeki fiziksel konumu
- Condition1: Ana yol veya tren yoluna yakınlık
- Condition2: Ana yola veya demiryoluna yakınlık (eğer ikinci bir yer varsa)
- BldgType: Konut tipi
- HouseStyle: Konut sitili
- OverallQual: Genel malzeme ve bitiş kalitesi
- OverallCond: Genel durum değerlendirmesi
- YearBuilt: Orijinal yapım tarihi
- YearRemodAdd: Yeniden düzenleme tarihi
- RoofStyle: Çatı tipi
- RoofMatl: Çatı malzemesi
- Exterior1st: Evdeki dış kaplama
- Exterior2nd: Evdeki dış kaplama (birden fazla malzeme varsa)
- MasVnrType: Duvar kaplama türü
- MasVnrArea: Kare ayaklı duvar kaplama alanı
- ExterQual: Dış malzeme kalitesi
- ExterCond: Malzemenin dışta mevcut durumu
- Foundation: Vakıf tipi
- BsmtQual: Bodrumun yüksekliği
- BsmtCond: Bodrum katının genel durumu
- BsmtExposure: Yürüyüş veya bahçe katı bodrum duvarları
- BsmtFinType1: Bodrum bitmiş alanının kalitesi
- BsmtFinSF1: Tip 1 bitmiş alanın metre karesi
- BsmtFinType2: İkinci bitmiş alanın kalitesi (varsa)
- BsmtFinSF2: Tip 2 bitmiş alanın metre karesi
- BsmtUnfSF: Bodrumun bitmemiş alanın metre karesi
- TotalBsmtSF: Bodrum alanının toplam metre karesi
- Heating: Isıtma tipi
- HeatingQC: Isıtma kalitesi ve durumu
- CentralAir: Merkezi klima
- Electrical: elektrik sistemi
- 1stFlrSF: Birinci Kat metre kare alanı
- 2ndFlrSF: İkinci kat metre kare alanı
- LowQualFinSF: Düşük kaliteli bitmiş alanlar (tüm katlar)
- GrLivArea: Üstü (zemin) oturma alanı metre karesi
- BsmtFullBath: Bodrum katındaki tam banyolar
- BsmtHalfBath: Bodrum katındaki yarım banyolar
- FullBath: Üst katlardaki tam banyolar
- HalfBath: Üst katlardaki yarım banyolar
- BedroomAbvGr: Bodrum seviyesinin üstünde yatak odası sayısı
- KitchenAbvGr: Bodrum seviyesinin üstünde mutfak Sayısı
- KitchenQual: Mutfak kalitesi
- TotRmsAbvGrd: Üst katlardaki toplam oda (banyo içermez)
- Functional: Ev işlevselliği değerlendirmesi
- Fireplaces: Şömineler
- FireplaceQu: Şömine kalitesi
- Garage Türü: Garaj yeri
- GarageYrBlt: Garajın yapım yılı
- GarageFinish: Garajın iç yüzeyi
- GarageCars: Araç kapasitesi
- GarageArea: Garajın alanı
- GarageQual: Garaj kalitesi
- GarageCond: Garaj durumu
- PavedDrive: Garajla yol arasındaki yol
- WoodDeckSF: Ayaklı ahşap güverte alanı
- OpenPorchSF: Kapı önündeki açık veranda alanı
- EnclosedPorch: Kapı önündeki kapalı veranda alan
- 3SsPorch: Üç mevsim veranda alanı
- ScreenPorch: Veranda örtü alanı
- PoolArea: Havuzun metre kare alanı
- PoolQC: Havuz kalitesi
- Fence: Çit kalitesi
- MiscFeature: Diğer kategorilerde bulunmayan özellikler
- MiscVal: Çeşitli özelliklerin değeri
- MoSold: Satıldığı ay
- YrSold: Satıldığı yıl
- SaleType: Satış Türü
- SaleCondition: Satış Durumu


## Script Modes:
```
--no-debug: Runs full script
--no-tuning: Runs script without tuning
--model-history: Shows models parameters and RMSE scores
```

## Make commands:

To run script with make commands first install MakeFile 
application on your IDE then run like:

`make command`
```
run: Runs script with debug mode

run_no_tuning: Runs script without tuning

run_debug: Runs script with debug mode

run_test: Runs test script

req: Creates requirements.txt

install: Installs requirements.txt

models: Shows models parameters and RMSE scores
```


# Files & Scripts

- *data/train.csv* - Training set
- *data/test.csv* - Test set
- *scripts/config.py*
- *scripts/data_preprocessing.py*
- *scripts/feature_engineering.py*
- *scripts/helper_functions.py*
- *scripts/predict.py*
- *scripts/train.py*
- *scripts/model_history.py*


## Outputs
- *pickles/models*
- *pickles/test_dataframe.pkl*
- *pickles/train_dataframe.pkl*
- *submission/*
- model_info_data.json


## Libraries Used

```
pandas
numpy
matplotlib
sklearn
scipy
pickle
catboost
lightgbm 
xgboost
mlxtend.regressor # This is for stacking part, works well with sklearn and others...
```

## Author

- Oğuz Han Erdoğan - [oguzerdo](https://github.com/oguzerdo)
