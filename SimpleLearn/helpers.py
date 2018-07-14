"""

    Author: Jun Wang
    Version: 1.0
    Project Name: simple-learn
    Created Date: 7/13/18
    Updated Date: 7/13/18
    Description: This is the helper function for the packages

"""
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


###########################################################################
# Run basic model
###########################################################################


def run_model(clf, x_train, y_train, x_test, y_test, label_list):
    """
    Run the model and print the performance based on the input
    :param clf: The given classifier
    :param x_train: The training data
    :param y_train: The label for the training data
    :param x_test: The test data
    :param y_test: The label for the test data
    :param label_list: The set for labels
    :return: The fitted classifier after training
    """
    clf.fit(x_train, y_train)

    y_train_predict = clf.predict(x_train)
    print "confusion matrix for training set: "
    print confusion_matrix(y_train, y_train_predict, labels=label_list)
    print "---------------------------------------"

    train_precision, train_recall, train_fscore, _ = precision_recall_fscore_support(y_train, y_train_predict,
                                                                                     average='macro')
    print "precision, recall and fscore of training set: "
    print train_precision, train_recall, train_fscore
    print "---------------------------------------"

    train_score = clf.score(x_train, y_train)
    print "training score: ", train_score

    print "#######################################"

    y_test_predict = clf.predict(x_test)
    print "confusion matrix for test set: "
    print confusion_matrix(y_test, y_test_predict, labels=label_list)
    print "---------------------------------------"

    test_precision, test_recall, test_fscore, _ = precision_recall_fscore_support(y_test, y_test_predict,
                                                                                  average='macro')
    print "precision, recall and fscore of training set: "
    print test_precision, test_recall, test_fscore
    print "---------------------------------------"

    test_score = clf.score(x_test, y_test)
    print "test score: ", test_score

    return clf


###########################################################################
# Feature selection by ANOVA
###########################################################################


def run_with_feature_selection(model, feature_num, _x_train_scaled, _y_train, _x_test_scaled, _y_test):
    """
    Train and predict the model with the given number of features for the input data
    :param model: The given classifier
    :param feature_num: The number of features
    :param _x_train_scaled: The training data
    :param _y_train: The label of the training data
    :param _x_test_scaled: The test data
    :param _y_test: The label of the test data
    :return: The f-score and the accuracy for the training set and test set
    """
    # Create an SelectKBest object to select features with two best ANOVA F-Values
    fvalue_selector = SelectKBest(f_classif, k=feature_num)

    # Apply the SelectKBest object to the features and target
    x_train_kbest = fvalue_selector.fit_transform(_x_train_scaled, _y_train)
    selected_features = fvalue_selector.get_support()

    x_test_kbest = _x_test_scaled[:, selected_features]

    model.fit(x_train_kbest, _y_train)

    y_train_predict = model.predict(x_train_kbest)

    train_precision, train_recall, train_fscore, _ = precision_recall_fscore_support(_y_train, y_train_predict,
                                                                                     average='macro')
    train_accuracy = model.score(x_train_kbest, _y_train)

    y_test_predict = model.predict(x_test_kbest)

    test_precision, test_recall, test_fscore, _ = precision_recall_fscore_support(_y_test, y_test_predict,
                                                                                  average='macro')
    test_accuracy = model.score(x_test_kbest, _y_test)

    return train_fscore, test_fscore, train_accuracy, test_accuracy


def plot_score_with_feature_selection(_features, model, _x_train_scaled, _y_train, _x_test_scaled, _y_test):
    """
    Plot the f-score with different number of features according to the training set
    :param _features: The list of features
    :param model: The given classifier
    :param _x_train_scaled: The training set
    :param _y_train: The label of the training set
    :param _x_test_scaled: The test set
    :param _y_test: The label of the test set
    :return: None
    """
    train_fscores = []
    test_fscores = []
    k_list = range(5, len(_features) + 1)
    for k in k_list:
        print k
        train_fscore, test_fscore, train_accuracy, test_accuracy = run_with_feature_selection(model, k,  _x_train_scaled, _y_train, _x_test_scaled, _y_test)
        train_fscores.append(train_fscore)
        test_fscores.append(test_fscore)
    plt.figure()
    plt.plot(k_list, train_fscores, 'bo-', label='fscore of training set')
    plt.plot(k_list, test_fscores, 'ro-', label='fscore of test set')
    plt.legend()
    plt.show()


###########################################################################
# Grid search for parameter tuning
###########################################################################


def run_gridsearchcv(dataset, param_grid, fold_num, model):
    """
    Run grid search for the given classifier and the parameter list
    :param dataset: The given dataset
    :param param_grid: The parameter list
    :param fold_num: The number of fold for cross validation
    :param model: The given classifier
    :return: None
    """
    # get data
    _X_train, _y_train, _X_test, _y_test = dataset

    # grid search
    gs = GridSearchCV(estimator=model,
                      param_grid=param_grid,
                      scoring='f1_macro',
                      n_jobs=-1,
                      cv=StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=0).split(_X_train, _y_train),
                      verbose=1,
                      refit=True,
                      pre_dispatch='2*n_jobs')
    # run gridearch
    gs.fit(_X_train, _y_train)

    # result
    print('Best GS Score %.9f' % gs.best_score_)
    print('best GS Params %s' % gs.best_params_)

    # prediction on the training set
    y_train_pred = gs.predict(_X_train)
    train_precision, train_recall, train_fscore, _ = precision_recall_fscore_support(_y_train, y_train_pred,
                                                                                     average='macro')
    print('Train fscore: %.9f' % train_fscore)

    # evaluation on the test set
    y_test_pred = gs.predict(_X_test)
    test_precision, test_recall, test_fscore, _ = precision_recall_fscore_support(_y_test, y_test_pred, average='macro')
    print('Test f-score: %.9f' % test_fscore)