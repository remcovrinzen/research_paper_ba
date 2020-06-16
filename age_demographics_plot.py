import pandas as pd
import os
import seaborn_plots as splot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def create_age_demographics_example_plot(age_gender_bkts):
    usa_demographic = age_gender_bkts[age_gender_bkts['country_destination'] == 'US']

    female_demographic = usa_demographic[usa_demographic['gender'] == 'female']

    male_demographic = usa_demographic[usa_demographic['gender'] == 'male']

    sorted_female_demographic = female_demographic.sort_values(
        by=['lower_bound_age_bucket'])

    sorted_male_demographic = male_demographic.sort_values(
        by=['lower_bound_age_bucket'])

    male_demographic = sorted_male_demographic[[
        'age_bucket', 'population_in_thousands']]
    female_demographic = sorted_female_demographic[[
        'age_bucket', 'population_in_thousands']]

    female_demographic = female_demographic.set_index(['age_bucket'])
    male_demographic = male_demographic.set_index(['age_bucket'])

    ind = np.arange(female_demographic.shape[0])
    sns.set(style="whitegrid", font_scale=1.5)
    ax = plt.subplot(111)

    width = 0.3

    ax.bar(ind, male_demographic['population_in_thousands'].values,
           width=width, label="Male")
    ax.bar(ind + width, female_demographic['population_in_thousands'].values, width=width,
           label="Female")

    # ax.set_title("Age gender demographics for USA")
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(female_demographic.index.values)
    plt.legend()
    plt.xlabel('Age bucket')
    plt.ylabel('Population (x1000)')

    sns.despine()

    plt.show()


age_demographics = pd.DataFrame(pd.read_csv(
    os.getcwd() + "/dataframes/rewritten_originals/age_gender_bkts_rewritten.csv"))

create_age_demographics_example_plot(age_demographics)
