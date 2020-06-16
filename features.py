import pandas as pd
import os
import numpy as np
from sklearn.externals import joblib
import pandas_helpers as ph
import psutil


class Features(object):
    def __init__(self):
        self.load_original_data()
        self.load_cleaned_data()
        self.general_setup_for_session_features()
        self.create_session_features_part_2()

    def load_original_data(self):
        data_path = os.getcwd() + "/data/"

        self.sessions = pd.DataFrame(pd.read_csv(
            data_path + "sessions.csv"))
        self.train_users = pd.DataFrame(
            pd.read_csv(data_path + "train_users_2.csv"))
        self.test_users = pd.DataFrame(
            pd.read_csv(data_path + "test_users.csv"))

        self.test_ids = self.test_users['id'].values

        self.old_df_all = pd.concat(
            (self.train_users, self.test_users), axis=0, ignore_index=True)

    def load_cleaned_data(self):
        data_path = os.getcwd() + "/dataframes/full_model_with_sessions_partially/"

        self.x_train = pd.DataFrame(
            pd.read_csv(data_path + "train_with_partially_sessions.csv", index_col=0))
        self.x_test = pd.DataFrame(
            pd.read_csv(data_path + "test_with_partially_sessions.csv", index_col=0))

        # self.y_train_benchmark = np.loadtxt(
        # data_path + "y_values_full_model.txt")
        # self.y_encoder = joblib.load(data_path + "y_encoder.pkl")

        self.test_ids = self.test_users['id'].values

        self.df_all = pd.concat(
            (self.x_train, self.x_test), axis=0, ignore_index=True)

    def general_setup_for_session_features(self):
        self.sessions = self.sessions.rename(columns={'user_id': 'id'})
        session_user_ids = self.sessions['id'].unique()
        self.sessions_summarized = pd.DataFrame(
            session_user_ids, columns=['id'])

        self.sessions_summarized = self.sessions_summarized.set_index('id')

        self.sessions['macro_event_string'] = self.sessions['action'].map(
            str) + self.sessions['action_type'].map(str) + self.sessions['action_detail'].map(str)

    def create_session_features_part_2(self):
        self.create_macro_event_sums()

        self.df_all = self.df_all.set_index(self.old_df_all['id'])
        self.df_all = self.df_all.join(self.sessions_summarized)
        ph.create_summary_file(self.df_all)
        self.df_all = self.df_all.fillna(-1)
        self.df_all = self.df_all.reset_index()
        self.train = self.df_all.loc[~self.df_all['id'].isin(
            self.test_ids)]
        self.test = self.df_all.loc[self.df_all['id'].isin(
            self.test_ids)]
        self.train = self.train.drop(['id'], axis=1)
        self.test = self.test.drop(['id'], axis=1)
        self.train.to_csv(os.getcwd() + "/train.csv")
        self.test.to_csv(os.getcwd() + "/test.csv")

    def create_macro_event_sums(self):
        macro_dummys = pd.get_dummies(self.sessions['macro_event_string'])

        process = psutil.Process(os.getpid())
        print(process.memory_info().rss / 1000000)
        macro_dummys['id'] = self.sessions['id']
        macro_events_per_user = macro_dummys.groupby('id').sum()

        self.sessions_summarized = self.sessions_summarized.join(
            macro_events_per_user)
