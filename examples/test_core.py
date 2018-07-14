"""

    Author: Jun Wang
    Version: 1.0
    Project Name: simple-learn
    Created Date: 7/13/18
    Updated Date:
    Description:

"""
from sklearn.linear_model import LogisticRegression

import SimpleLearn.core as sl

if __name__=="__main__":
    data = sl.Dataset()
    data.read_data('../examples/data', "phone_no")
    clf = LogisticRegression(penalty='l1')
    data.split_dataset(test_size=0.2, random_state=11, shuffle=True, is_stratify=True)
    data.standard_scaling()
    data.run_model(clf)
    # preprocessor = sl.Preprocesser()
    # preprocessor.set_base_model(clf)
    # preprocessor.plot_performance_with_features(data)
    data.select_K_features(10)
    print data.X_train.shape
    print data.preprocess_procedure
    # data.reset_dataset()
    # print data.X_train.shape
    print "#######################--------------------------------############################"
    data.run_model(clf)
    print "#######################--------------------------------############################"
    param_grid = [
        {'penalty': ['l1', 'l2'],
         'C': [0.1, 1, 10]}
    ]
    classifier = sl.Classifier()
    classifier.set_clf(clf)
    classifier.set_parameters(param_grid)
    classifier.grid_search(data, 5)