import pandas as pd
import os
import numpy as np
import pandas_helpers as ph
import seaborn_plots as splot
import matplotlib.pyplot as plt


class Eda(object):
    def __init__(self):
        self.import_data()
        self.do_eda()

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

    def do_eda(self):
        # ph.create_summary_file(self.age_gender_bkts)
        # ph.create_summary_file(self.countries)
        ph.create_summary_file(self.train_users)
        # ph.create_summary_file(self.test_users)
        # ph.create_summary_file(self.sessions)

        # self.look_into_missing_values()
        # self.look_into_sessions_actions()
        # self.look_into_overlap_user_ids()
        # self.look_into_first_sec_elapsed_per_user_id()
        # self.create_age_demographics_example_plot()

    def look_into_missing_values(self):
        pass
        # self.look_into_train_set()
        # self.look_into_missing_sessions()

    def look_into_train_set(self):
        self.look_into_date_first_booking()
        self.look_into_first_affiliate_tracked()
        self.look_into_age()
        self.look_into_unknown_values()

    def look_into_date_first_booking(self):
        missing_date_first_booking_rows = self.train_users[
            self.train_users['date_first_booking'].isnull(
            )]
        nan_rows = missing_date_first_booking_rows.shape[0]

        corresponding_country_locations_percentages = missing_date_first_booking_rows[
            'country_destination'].value_counts() / nan_rows * 100

    def look_into_first_affiliate_tracked(self):
        missing_date_first_affiliate_tracked = self.train_users[self.train_users['first_affiliate_tracked'].isnull(
        )]

        nan_rows = missing_date_first_affiliate_tracked.shape[0]

        corresponding_device_percentages = missing_date_first_affiliate_tracked['first_device_type'].value_counts(
        ) / nan_rows * 100
        corresponding_browser_percentages = missing_date_first_affiliate_tracked['first_browser'].value_counts(
        ) / nan_rows * 100

        corresponding_affiliate_channel_percentages = missing_date_first_affiliate_tracked['affiliate_channel'].value_counts(
        ) / nan_rows * 100

    def look_into_age(self):
        clean_age_data = self.train_users['age'].dropna()

    def look_into_unknown_values(self):
        vars_with_unknown = ['gender', 'first_device_type', 'first_browser']

        for var in vars_with_unknown:
            value = '-unknown-'

            if var == 'first_device_type':
                value = 'Other/Unknown'

            number_of_rows = ph.get_number_of_rows(self.train_users)

            percentage_unknown = ph.get_number_of_rows(
                self.train_users[self.train_users[var] == value]) / number_of_rows * 100

            percentage_unknown = ph.get_number_of_rows(
                self.train_users[self.train_users[var] == value]) / number_of_rows * 100

    def look_into_sessions_actions(self):
        self.sessions = self.sessions.fillna('-unknown-')
        print(self.sessions.groupby(
            ['action', 'action_type', 'action_detail']).nunique())

    def look_into_missing_sessions(self):
        missing_action = self.sessions[self.sessions['action'].isnull()]

        nan_rows = ph.get_number_of_rows(missing_action)

        corresponding_secs_elapsed_percentages = missing_action['secs_elapsed'].value_counts(
        ) / nan_rows * 100

        missing_action_type = self.sessions[self.sessions['action_type'].isnull(
        )]

        nan_rows = ph.get_number_of_rows(missing_action_type)

        corresponding_action_detail_percentages = missing_action_type['action_detail'].value_counts(
        ) / nan_rows * 100

    def look_into_overlap_user_ids(self):
        ids_in_sessions = self.sessions['user_id'].unique()
        train_users_ids = self.train_users['id']
        test_users_ids = self.test_users['id']
        percentage_values_train = train_users_ids.isin(
            ids_in_sessions).value_counts() / ph.get_number_of_rows(train_users_ids) * 100

        percentage_values_test = test_users_ids.isin(
            ids_in_sessions).value_counts() / ph.get_number_of_rows(test_users_ids) * 100

    def create_benchmark_sets(self):
        pass

    def create_age_demographics_example_plot(self):
        usa_demographic = self.age_gender_bkts[self.age_gender_bkts['country_destination'] == 'US']

        female_demographic = usa_demographic[usa_demographic['gender'] == 'female'][[
            'age_bucket', 'population_in_thousands']]

        male_demographic = usa_demographic[usa_demographic['gender'] == 'male'][[
            'age_bucket', 'population_in_thousands']]

        female_demographic = female_demographic.set_index(['age_bucket'])
        male_demographic = male_demographic.set_index(['age_bucket'])

        female_demographic = female_demographic.sort_index()

        male_demographic.plot(kind="bar", width=0.4,
                              position=0, label="Male")
        female_demographic.plot(kind="bar", width=0.4,
                                position=1, label="Female")
        plt.legend()
        plt.xlabel('Age bucket')
        plt.ylabel('Population (x1000)')
        plt.show()
