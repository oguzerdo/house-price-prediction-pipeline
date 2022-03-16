import pickle as pickle

from scripts.helper_functions import *


#######################################
# Uploading Results
#######################################
def predict(final_model, selected_features):
    pickle_dir = os.getcwd() + '/outputs/pickles/'
    test_df = pickle.load(open(pickle_dir + 'test_dataframe.pkl', 'rb'))

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

