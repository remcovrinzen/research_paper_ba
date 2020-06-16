import os
import pandas as pd
import pandas_helpers as ph
import seaborn_plots as sb
from sklearn.externals import joblib

y_encoder = joblib.load(os.getcwd() + "/dataframes/benchmark/y_encoder.pkl")

# print(y_encoder)
print(list(y_encoder.classes_))
# y_encoder.transform(["tokyo", "tokyo", "paris"])
print(list(y_encoder.inverse_transform([0])))
