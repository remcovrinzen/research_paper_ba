import pandas as pd
import os
import sklearn.preprocessing
import ml_helpers as mh
import numpy as np
from sklearn.externals import joblib
import pandas_helpers as ph


class Cleaning(object):
    def __init__(self):
        self.import_data()
        self.dataframes_dir = os.getcwd() + "/dataframes/"

        self.define_categorical_values()
        # self.create_benchmark_sets()
        self.create_full_model_sets()
        self.save_full_models()

    def import_data(self):
        data_path = os.getcwd() + "/data/"

        self.age_gender_bkts = pd.DataFrame(
            pd.read_csv(data_path + "age_gender_bkts.csv"))
        self.countries = pd.DataFrame(pd.read_csv(data_path + "countries.csv"))
        self.sessions = pd.DataFrame(pd.read_csv(data_path + "sessions.csv"))
        self.train_users = pd.DataFrame(
            pd.read_csv(data_path + "train_users_2.csv"))
        self.test_users = pd.DataFrame(
            pd.read_csv(data_path + "test_users.csv"))

        self.test_ids = self.test_users['id'].values

        self.old_df_all = pd.concat(
            (self.train_users, self.test_users), axis=0, ignore_index=True)

    def create_benchmark_sets(self):
        benchmark_dir = self.dataframes_dir + "benchmark/"

        if not os.path.exists(benchmark_dir):
            os.makedirs(benchmark_dir)

        self.test_benchmark_ids = self.test_users['id'].values
        self.y_train_benchmark = self.train_users['country_destination'].values

        self.x_train_benchmark = self.drop_useless_columns(
            self.train_users, ['date_first_booking', 'country_destination'])

        self.test_benchmark = self.drop_useless_columns(
            self.test_users, ['date_first_booking'])

        number_of_rows_train = ph.get_number_of_rows(self.y_train_benchmark)

        df_all = pd.concat(
            (self.x_train_benchmark, self.test_benchmark), axis=0, ignore_index=True)

        df_all = self.fill_nan_df_with_negative(df_all)

        df_all = mh.transform_values_into_categorical(
            df_all, self.users_categorical)

        df_all = self.date_account_created_to_day_month_year(df_all)

        df_all = self.timestamp_first_active_to_day_month_year(df_all)

        self.y_train_benchmark, self.label_encoder_benchmark = self.create_labels_from_y_values(
            self.y_train_benchmark)

        self.x_train_benchmark = df_all.loc[~df_all['id'].isin(
            self.test_benchmark_ids)]
        self.test_benchmark = df_all.loc[df_all['id'].isin(
            self.test_benchmark_ids)]

        self.x_train_benchmark = self.drop_useless_columns(
            self.x_train_benchmark, ['id'])

        self.test_benchmark = self.drop_useless_columns(
            self.test_benchmark, ['id']
        )

        self.x_train_benchmark.to_csv(benchmark_dir + "x_train_benchmark.csv")
        self.test_benchmark.to_csv(benchmark_dir + "test_benchmark.csv")
        np.savetxt(benchmark_dir + "y_values_benchmark.txt",
                   self.y_train_benchmark)
        joblib.dump(self.label_encoder_benchmark,
                    benchmark_dir + "y_encoder.pkl")

    def drop_useless_columns(self, df, column_array):
        result = df.drop(column_array, axis=1)
        return result

    def fill_nan_df_with_negative(self, df):
        return df.fillna(-1)

    def create_labels_from_y_values(self, y_values):
        return mh.create_labels_from_y_values(y_values)

    def define_categorical_values(self):
        self.users_categorical = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel',
                                  'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']

    def date_account_created_to_day_month_year(self, df):
        dac = np.vstack(df.date_account_created.astype(
            str).apply(lambda x: list(map(int, x.split('-')))).values)
        df['dac_year'] = dac[:, 0]
        df['dac_mounth'] = dac[:, 1]
        df['dac_day'] = dac[:, 2]
        df = df.drop(['date_account_created'], axis=1)
        return df

    def timestamp_first_active_to_day_month_year(self, df):
        tfa = np.vstack(df.timestamp_first_active.astype(str).apply(lambda x: list(
            map(int, [x[:4], x[4:6], x[6:8], x[8:10], x[10:12], x[12:14]]))).values)
        df['tfa_year'] = tfa[:, 0]
        df['tfa_month'] = tfa[:, 1]
        df['tfa_day'] = tfa[:, 2]
        df = df.drop(['timestamp_first_active'], axis=1)
        return df

    def basic_cleaning(self):
        self.full_model_dir = self.dataframes_dir + "full_model/"

        if not os.path.exists(self.full_model_dir):
            os.makedirs(self.full_model_dir)

        self.test_full_model_ids = self.test_users['id'].values
        self.y_train_full_model = self.train_users['country_destination'].values

        self.x_train_full_model = self.drop_useless_columns(
            self.train_users, ['date_first_booking', 'country_destination'])

        self.test_full_model = self.drop_useless_columns(
            self.test_users, ['date_first_booking'])

        number_of_rows_train = ph.get_number_of_rows(self.y_train_full_model)

        self.df_all = pd.concat(
            (self.x_train_full_model, self.test_full_model), axis=0, ignore_index=True)

        self.df_all = self.fill_nan_df_with_negative(self.df_all)

        self.df_all = mh.transform_values_into_categorical(
            self.df_all, self.users_categorical)

        self.df_all = self.timestamp_first_active_to_day_month_year(
            self.df_all)

        self.df_all = self.date_account_created_to_day_month_year(self.df_all)

        self.y_train_full_model, self.label_encoder_full_model = self.create_labels_from_y_values(
            self.y_train_full_model)

    def create_full_model_sets(self):
        self.basic_cleaning()
        self.create_features()

    def create_features(self):
        self.create_countries_features()
        self.create_age_demographics_features()

    def create_countries_features(self):
        self.convert_language_strings()
        self.create_levenhein_dist_per_country()

    def create_levenhein_dist_per_country(self):
        destination_languages = self.countries['destination_language '].unique(
        )

        for language in destination_languages:
            self.df_all['ld_to_' +
                        language] = self.old_df_all['language'].apply(self.determine_levenhein_distance, args=(language,))

    def determine_levenhein_distance(self, row, language):
        user_language = row

        if not user_language == 'en' and not language == 'en':  # ENGLISH
            return -1
        else:
            if user_language == 'en' and language == 'en':
                return 0
            elif user_language == 'en':
                target_language = language
            else:
                target_language = user_language

            lh_distances = self.countries[self.countries['destination_language ']
                                          == target_language]['language_levenshtein_distance'].values

            if len(lh_distances) == 0:
                return -1
            else:
                return lh_distances[0]

    def convert_language_strings(self):
        converter = {'eng': 'en', 'deu': 'de',
                     'spa': 'es', 'fra': 'fr', 'ita': 'it', 'nld': 'nl', 'por': 'pt'}

        for language in converter:
            self.countries['destination_language '].loc[self.countries['destination_language ']
                                                        == language] = converter[language]

    def save_full_models(self):
        self.x_train_full_model = self.df_all.loc[~self.df_all['id'].isin(
            self.test_full_model_ids)]
        self.test_full_model = self.df_all.loc[self.df_all['id'].isin(
            self.test_full_model_ids)]

        self.x_train_full_model = self.drop_useless_columns(
            self.x_train_full_model, ['id'])

        self.test_full_model = self.drop_useless_columns(
            self.test_full_model, ['id']
        )

        self.x_train_full_model.to_csv(
            self.full_model_dir + "x_train_full_model.csv")
        self.test_full_model.to_csv(
            self.full_model_dir + "test_full_model.csv")
        np.savetxt(self.full_model_dir + "y_values_full_model.txt",
                   self.y_train_full_model)
        joblib.dump(self.label_encoder_full_model,
                    self.full_model_dir + "y_encoder.pkl")

    def create_age_demographics_features(self):
        self.rewrite_demographics()
        self.create_percentage_same_age_to_total()
        print(self.df_all.head())

    def rewrite_demographics(self):
        self.age_gender_bkts['upper_bound_age_bucket'] = self.age_gender_bkts['age_bucket'].apply(
            self.get_upper_bound_age_bucket)

        self.age_gender_bkts['lower_bound_age_bucket'] = self.age_gender_bkts['age_bucket'].apply(
            self.get_lower_bound_age_bucket)

        self.age_gender_bkts.to_csv(
            os.getcwd() + "/dataframes/rewritten_originals/age_gender_bkts_rewritten.csv")

    def get_upper_bound_age_bucket(self, row):
        age_bucket = row

        splitted_age_bucket = age_bucket.split('-')

        if len(splitted_age_bucket) == 1:
            return np.Inf
        else:
            return int(age_bucket.split('-')[1])

    def get_lower_bound_age_bucket(self, row):
        age_bucket = row

        splitted_age_bucket = age_bucket.split('-')

        if len(splitted_age_bucket) == 1:
            return 100
        else:
            return int(age_bucket.split('-')[0])

    def create_percentage_same_age_to_total(self):
        unique_destination = self.age_gender_bkts['country_destination'].unique(
        )

        for country in unique_destination:
            self.df_all['fraction_same_age_in_' +
                        country] = self.df_all['age'].apply(self.get_fraction_same_age_in_country, args=(country,))

    def get_fraction_same_age_in_country(self, row, country):
        user_age = int(row)

        if user_age == -1:
            return -1

        total_country = self.age_gender_bkts[self.age_gender_bkts['country_destination']
                                             == country]['population_in_thousands'].sum()

        same_age_in_country = self.age_gender_bkts["population_in_thousands"][(
            self.age_gender_bkts["lower_bound_age_bucket"] <= user_age) & (self.age_gender_bkts["upper_bound_age_bucket"] >= user_age)].values[0]

        return same_age_in_country / total_country

    def create_session_features(self):
        self.groupby_statistics()

        self.sessions_summarized['repeated_actions_count'] = self.sessions_summarized['macro_event_string_count'] - \
            self.sessions_summarized['macro_event_string_nunique']

        self.create_last_event()

        self.df_all = self.df_all.set_index(self.old_df_all['id'])
        self.df_all = self.df_all.join(self.sessions_summarized)
        ph.create_summary_file(self.df_all)
        self.df_all = self.df_all.fillna(-1)
        self.df_all = self.df_all.reset_index()
        self.x_train = self.df_all.loc[~self.df_all['id'].isin(
            self.test_ids)]
        self.x_test = self.df_all.loc[self.df_all['id'].isin(
            self.test_ids)]
        self.x_train = self.x_train.drop(['id'], axis=1)
        self.x_test = self.x_test.drop(['id'], axis=1)
        self.x_train.to_csv(os.getcwd() + "/train.csv")
        self.x_test.to_csv(os.getcwd() + "/test.csv")

    def groupby_statistics(self):
        statistics = {'secs_elapsed': [
            'sum', 'mean', 'min', 'max', 'last'], 'action': ['nunique'], 'action_type': ['nunique'], 'action_detail': ['nunique'], 'device_type': ['nunique'], 'macro_event_string': ['count', 'nunique']}

        info_per_user = self.sessions.groupby(
            'id').agg(statistics)

        info_per_user.columns = ["_".join(x)
                                 for x in info_per_user.columns.ravel()]
        self.sessions_summarized = self.sessions_summarized.join(
            info_per_user)

    def create_last_event(self):
        last_macro_event_per_user = self.sessions.groupby(
            'id')['macro_event_string'].last()
        last_event_dummys = pd.get_dummies(
            last_macro_event_per_user, prefix='last_event')
        self.sessions_summarized = self.sessions_summarized.join(
            last_event_dummys)

    def general_setup_for_session_features(self):
        self.sessions = self.sessions.rename(columns={'user_id': 'id'})
        session_user_ids = self.sessions['id'].unique()
        self.sessions_summarized = pd.DataFrame(
            session_user_ids, columns=['id'])

        self.sessions_summarized = self.sessions_summarized.set_index('id')

        self.sessions['macro_event_string'] = self.sessions['action'].map(
            str) + self.sessions['action_type'].map(str) + self.sessions['action_detail'].map(str)
