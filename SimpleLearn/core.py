"""

    Author: Jun Wang
    Version: 1.0
    Project Name: simple-learn
    Created Date: 7/13/18
    Updated Date:
    Description:

"""

import os
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression


import helpers

class Dataset:
    def __init__(self):
        self.dataset = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.features = None
        self.labels = None
        self._split_parameter = None
        self.preprocess_procedure = []

    def read_data(self, path, id_column_name):
        buffer = []
        labels = []
        for filename in os.listdir(path):
            current_data = pd.read_csv(path + '/' + filename, na_values=['\N'])
            current_label = filename.strip().split('.')[0]
            current_data['label'] = current_label
            labels.append(current_label)
            buffer.append(current_data)
        dataset = pd.concat(buffer)
        self.dataset = dataset.dropna()
        self.labels = labels
        features = list(self.dataset.columns)
        features.remove(id_column_name)
        features.remove('label')
        self.features = features

    def split_dataset(self, test_size, random_state = 1, shuffle = True, is_stratify=True):
        X = self.dataset[self.features].as_matrix()
        Y = self.dataset['label'].as_matrix()
        self._split_parameter = [test_size, random_state, shuffle, is_stratify]
        if is_stratify:
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=Y)
        else:
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state, shuffle=shuffle)

    def standard_scaling(self):
        # get the parameter of scaling from training set
        scaler = preprocessing.StandardScaler()
        scale_param = scaler.fit(self.X_train)
        # scaling the training set
        X_train_preprocessed = scaler.fit_transform(self.X_train, scale_param)
        # scaling the test set
        X_test_preprocessed = scaler.fit_transform(self.X_test, scale_param)

        self.X_train = X_train_preprocessed
        self.X_test = X_test_preprocessed
        self.preprocess_procedure.append("standard_scaling")
        return scale_param

    def select_K_features(self, feature_num):
        # Create an SelectKBest object to select features with two best ANOVA F-Values
        fvalue_selector = SelectKBest(f_classif, k=feature_num)

        # Apply the SelectKBest object to the features and target
        X_train_kbest = fvalue_selector.fit_transform(self.X_train, self.Y_train)
        selected_features = fvalue_selector.get_support()
        X_test_kbest = self.X_test[:, selected_features]
        self.X_train = X_train_kbest
        self.X_test = X_test_kbest
        self.preprocess_procedure.append("feature_selection")
        return selected_features

    def reset_dataset(self):
        self.split_dataset(test_size=self._split_parameter[0], random_state=self._split_parameter[1], shuffle=self._split_parameter[2], is_stratify=self._split_parameter[3])

    def run_model(self, clf):
        fitted_clf = helpers.run_model(clf, self.X_train, self.Y_train, self.X_test, self.Y_test, self.labels)
        return fitted_clf

class Preprocesser:
    def __init__(self):
        self.base_clf = None

    def set_base_model(self, clf):
        self.base_clf = clf

    def plot_performance_with_features(self, data):
        assert isinstance(data, Dataset), "Input should be the instance of the class Dataset"
        helpers.plot_score_with_feature_selection(data.features, self.base_clf, data.X_train, data.Y_train, data.X_test, data.Y_test)


class Classifier:
    def __init__(self):
        self.clf = None
        self.param_list = None

    def set_clf(self, clf):
        self.clf = clf

    def set_parameters(self, params):
        self.param_list = params

    def grid_search(self, data, fold_num):
        assert isinstance(data, Dataset), "The input variable data should be the instance of the class Dataset"
        dataset = (data.X_train, data.Y_train, data.X_test, data.Y_test)
        helpers.run_gridsearchcv(dataset, self.param_list, fold_num, self.clf)


if __name__=="__main__":
    data = Dataset()
    data.read_data('../examples/data', "phone_no")
    clf = LogisticRegression(penalty='l1')
    data.split_dataset(test_size=0.2, random_state=11, shuffle=True, is_stratify=True)
    data.standard_scaling()
    data.run_model(clf)
    preprocessor = Preprocesser()
    preprocessor.set_base_model(clf)
    preprocessor.plot_performance_with_features(data)
    data.select_K_features(10)
    print data.X_train.shape
    print data.preprocess_procedure
    data.reset_dataset()
    print data.X_train.shape
    print "#######################--------------------------------############################"
    # data.run_model(clf)
    print "#######################--------------------------------############################"
    # param_grid = [
    #     {'penalty': ['l1', 'l2'],
    #      'C': [0.1, 1, 10]}
    # ]
    # classifier = Classifier()
    # classifier.set_clf(clf)
    # classifier.set_parameters(param_grid)
    # classifier.grid_search(data, 5)