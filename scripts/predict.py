import pickle as pickle

from scripts.helper_functions import *


#######################################
# Uploading Results
#######################################
def predict(final_model):
    pickle_dir = os.getcwd() + '/outputs/pickles/'
    train_df = pickle.load(open(pickle_dir + 'train_dataframe.pkl', 'rb'))
    test_df = pickle.load(open(pickle_dir + 'test_dataframe.pkl', 'rb'))
    y = train_df['SalePrice']
    X = train_df.drop(["SalePrice", "Id"], axis=1)
    selected_features = feature_selection(X, y)

    ##### Predicting #####
    submission_df = pd.DataFrame()
    submission_df['Id'] = test_df["Id"].astype(int)
    y_pred_sub = final_model.predict(test_df[selected_features])
    submission_df['SalePrice'] = y_pred_sub
    curr_dir = os.getcwd()
    today = pd.to_datetime("today").strftime('%d-%m-%Y-%H-%M')
    os.makedirs("outputs/submission", exist_ok=True)
    result_dir = curr_dir + '/outputs/submission/'
    submission_df.to_csv(result_dir + f'submission{today}.csv', index=False)

