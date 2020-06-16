from sklearn.externals import joblib
from xgboost import plot_importance, plot_tree
import os
from matplotlib import pyplot
import pandas as pd
import operator

models_dir = os.getcwd() + "/models/"

model = joblib.load(models_dir + "/full_model.pkg")
train_frame = pd.DataFrame(pd.read_csv(
    os.getcwd() + "/dataframes/full_model/train.csv", index_col=0))

feature_names = train_frame.columns

mapper = {'f{0}'.format(i): v for i, v in enumerate(feature_names)}
features = ['f638', 'f729', 'f870', 'f256',
            'f348', 'f81', 'f168', 'f739', 'f159', 'f606']

# with open(os.getcwd() + "/feature_test_map.txt", 'w') as file:
#     for key in mapper:
#         file.write(key + " ")
#         file.write(mapper[key] + " ")
#         file.write(" q\n")

for key in features:
    print(key, mapper[key])
exit()
mapped = {mapper[k]: v for k, v in model.get_booster(
).get_score().items()}

# print(feature_names.values)
# model.get_booster().feature_names = feature_names.values
# print(model.get_booster().feature_names)

# top10 = dict(
#     sorted(mapped.items(), key=operator.itemgetter(1), reverse=True)[:10])
# # print(top10)
# # plot_importance(top10, color='green')
plot_tree(model)
# fig = pyplot.gcf()
# fig.set_size_inches(100, 50)

# pyplot.savefig(os.getcwd() + "/plots/test.png")
pyplot.show()
