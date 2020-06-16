import os
import pandas as pd
from sklearn.externals import joblib
import numpy as np
from xgboost import XGBClassifier
import xgboost
from sklearn.model_selection import StratifiedKFold
import math
import json


class Models(object):
    def __init__(self):
        self.models_dir = os.getcwd() + "/models/"

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        self.test_user_ids = pd.DataFrame(pd.read_csv(
            os.getcwd() + "/data/test_users.csv"))['id'].values

        # self.do_benchmark()
        # self.do_full_model()
        # self.do_k_fold_cross_validation(3)
        # self.load_full_frames()
        # self.optimize_model()
        self.do_tuned_model()

    def do_benchmark(self):
        self.load_benchmark_frames()
        self.create_benchmark_model()
        # self.benchmark_model = self.load_model("benchmark")
        predictions = self.create_predictions(
            self.test_benchmark, self.benchmark_model)
        self.generate_csv_file_with_5_best_predictions(
            predictions, self.y_encoder, "benchmark")

    def do_full_model(self):
        self.load_full_frames()
        self.create_full_model()
        predictions = self.create_predictions(
            self.x_test_full_model, self.full_model)
        self.generate_csv_file_with_5_best_predictions(
            predictions, self.y_encoder, "full_model")

    def create_benchmark_model(self):
        self.benchmark_model = self.train_model(
            self.x_train_benchmark, self.y_train_benchmark)
        self.save_model(self.benchmark_model, "benchmark")

    def create_full_model(self):
        self.full_model = self.train_model(
            self.x_train_full_model, self.y_train_full_model)
        self.save_model(self.full_model, "full_model")

    def load_benchmark_frames(self):
        benchmark_dir = os.getcwd() + "/dataframes/benchmark/"

        self.x_train_benchmark = pd.DataFrame(
            pd.read_csv(benchmark_dir + "x_train_benchmark.csv", index_col=0))
        self.test_benchmark = pd.DataFrame(
            pd.read_csv(benchmark_dir + "test_benchmark.csv", index_col=0))

        self.y_train_benchmark = np.loadtxt(
            benchmark_dir + "y_values_benchmark.txt")
        self.y_encoder = joblib.load(benchmark_dir + "y_encoder.pkl")

    def load_full_frames(self):
        full_model_dir = os.getcwd() + "/dataframes/full_model/"

        self.x_train_full_model = pd.DataFrame(pd.read_csv(
            full_model_dir + "/train.csv", index_col=0))

        self.x_test_full_model = pd.DataFrame(pd.read_csv(
            full_model_dir + "/test.csv", index_col=0))

        self.y_train_full_model = np.loadtxt(
            full_model_dir + "/y_values_full_model.txt")

        self.y_encoder = joblib.load(full_model_dir + "/y_encoder.pkl")

    def train_model(self, x_values, y_values, params):
        xgb = XGBClassifier(
            booster=params['booster'],
            eta=params['eta'],
            min_child_weight=params['min_child_weight'],
            max_depth=params['max_depth'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'])
        x = x_values.as_matrix()
        y = y_values
        return xgb.fit(X=x, y=y, eval_metric=["ndcg@5"])

    def create_predictions(self, x_predict, model):
        return model.predict_proba(x_predict.as_matrix())

    def generate_csv_file_with_5_best_predictions(self, predictions, le, model_name):
        solutions_dir = os.getcwd() + "/solutions/"

        if not os.path.exists(solutions_dir):
            os.makedirs(solutions_dir)

        user_ids = np.repeat(self.test_user_ids, 5)
        predicted_countries = []

        for i in range(len(predictions)):
            if len(predicted_countries) == 0:
                predicted_countries = le.inverse_transform(
                    np.argsort(predictions[i])[::-1])[:5].tolist()
            else:
                predicted_countries += le.inverse_transform(
                    np.argsort(predictions[i])[::-1])[:5].tolist()

        solution = pd.DataFrame(np.column_stack((user_ids, predicted_countries)),
                                columns=['id', 'country'])
        solution.to_csv(solutions_dir + model_name +
                        '_solution.csv', index=False)

    def save_model(self, model, model_name):
        joblib.dump(model, self.models_dir + model_name + ".pkg")

    def load_model(self, model_name):
        return joblib.load(self.models_dir + model_name + ".pkg")

    def do_k_fold_cross_validation(self, folds):
        self.load_full_frames()

        boosters = ["gbtree"]
        max_depths = [5]
        min_child_weights = [1]
        subsamples = [0.8]
        column_subsamples = [0.8]
        etas = [0.3]

        folder = StratifiedKFold(n_splits=folds)

        results = {}

        k_folds = folder.split(
            self.x_train_full_model, self.y_train_full_model)

        n = 1

        for train_index, test_index in k_folds:
            X_train, X_test = self.x_train_full_model.iloc[
                train_index], self.x_train_full_model.iloc[test_index]
            y_train, y_test = self.y_train_full_model[train_index], self.y_train_full_model[test_index]

            for max_depth in max_depths:
                for booster in boosters:
                    for eta in etas:
                        for min_child_weight in min_child_weights:
                            for subsample in subsamples:
                                for column_subsample in column_subsamples:
                                    parameters = {
                                        'booster': booster, 'eta': eta, 'max_depth': max_depth, 'min_child_weight': min_child_weight, 'subsample': subsample, 'colsample_bytree': column_subsample}

                                    if json.dumps(parameters) not in results.keys():
                                        results[json.dumps(parameters)] = [
                                            json.dumps(parameters)]

                                    print(n)

                                    model = self.train_model(
                                        X_train, y_train, parameters)
                                    predictions = self.create_predictions(
                                        X_test, model)

                                    nDCG = self.calculate_nDCG(
                                        predictions, y_test)

                                    results[json.dumps(
                                        parameters)].append(nDCG)

            n += 1

        with open(os.getcwd() + '/k_fold_summary.txt', 'w') as summary_file:
            for model in results:
                for line in results[model]:
                    summary_file.write(str(line) + "\n")
                summary_file.write("\n")

    def calculate_nDCG(self, predictions, true_values):
        le = self.y_encoder
        int_values = [int(x) for x in true_values]
        true_classes = le.inverse_transform(int_values)

        nDCGs = []

        for i in range(len(predictions)):
            prediction_array = le.inverse_transform(
                np.argsort(predictions[i])[::-1])[:5].tolist()

            true_class = true_classes[i]

            if true_class not in prediction_array:
                nDCGs.append(0)
            else:
                place = prediction_array.index(true_class)
                nDCG = 1 / math.log2(place + 1 + 1)  # 2^1-1=1
                nDCGs.append(nDCG)

        return np.mean(nDCGs)

    def optimize_model(self):
        xgb = XGBClassifier(
            objective="multi:softprob",
            learning_rate=0.1,
            num_class=12,
            n_estimators=1000,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            seed=27
        )

        cv_folds = 5
        early_stopping_rounds = 50

        xgb_param = xgb.get_xgb_params()

        x = self.x_train_full_model
        y = self.y_train_full_model
        xgtrain = xgboost.DMatrix(x.as_matrix(), y)
        cvresult = xgboost.cv(xgb_param, xgtrain, num_boost_round=xgb.get_params()['n_estimators'], nfold=cv_folds,
                              metrics='ndcg@5', early_stopping_rounds=early_stopping_rounds, stratified=True)
        joblib.dump(cvresult, os.getcwd() +
                    "/validation_result_n_estimators_first.pkg")

    def do_tuned_model(self):
        self.load_full_frames()

        parameters = {'booster': 'gbtree', 'eta': 0.3, 'max_depth': 5,
                      'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8}

        model = self.train_model(
            self.x_train_full_model, self.y_train_full_model, parameters)
        joblib.dump(model, os.getcwd() + "/models/full_model.pkg")

        predictions = self.create_predictions(
            self.x_test_full_model, model)
        self.generate_csv_file_with_5_best_predictions(
            predictions, self.y_encoder, "full_model")
