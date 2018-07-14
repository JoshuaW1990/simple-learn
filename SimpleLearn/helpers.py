"""

    Author: Jun Wang
    Version: 1.0
    Project Name: simple-learn
    Created Date: 7/13/18
    Updated Date:
    Description:

"""
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

###########################################################################
# Check basic model
###########################################################################


def run_model(clf, X_train, y_train, X_test, y_test, label_list):

    clf.fit(X_train, y_train)

    y_train_predict = clf.predict(X_train)
    print "confusion matrix for training set: "
    print confusion_matrix(y_train, y_train_predict, labels=label_list)
    print "---------------------------------------"

    train_precision, train_recall, train_fscore, _ = precision_recall_fscore_support(y_train, y_train_predict,
                                                                                     average='macro')
    print "precision, recall and fscore of training set: "
    print train_precision, train_recall, train_fscore
    print "---------------------------------------"

    train_score = clf.score(X_train, y_train)
    print "training score: ", train_score

    print "#######################################"

    y_test_predict = clf.predict(X_test)
    print "confusion matrix for test set: "
    print confusion_matrix(y_test, y_test_predict, labels=label_list)
    print "---------------------------------------"

    test_precision, test_recall, test_fscore, _ = precision_recall_fscore_support(y_test, y_test_predict,
                                                                                  average='macro')
    print "precision, recall and fscore of training set: "
    print test_precision, test_recall, test_fscore
    print "---------------------------------------"

    test_score = clf.score(X_test, y_test)
    print "test score: ", test_score

    return clf


###########################################################################
# Feature selection by ANOVA
###########################################################################


def run_with_feature_selection(model, feature_num, _X_train_scaled, _y_train, _X_test_scaled, _y_test):
    # Create an SelectKBest object to select features with two best ANOVA F-Values
    fvalue_selector = SelectKBest(f_classif, k=feature_num)

    # Apply the SelectKBest object to the features and target
    X_train_kbest = fvalue_selector.fit_transform(_X_train_scaled, _y_train)
    selected_features = fvalue_selector.get_support()

    X_test_kbest = _X_test_scaled[:, selected_features]

    model.fit(X_train_kbest, _y_train)

    y_train_predict = model.predict(X_train_kbest)

    train_precision, train_recall, train_fscore, _ = precision_recall_fscore_support(_y_train, y_train_predict,
                                                                                     average='macro')
    train_accuracy = model.score(X_train_kbest, _y_train)

    y_test_predict = model.predict(X_test_kbest)

    test_precision, test_recall, test_fscore, _ = precision_recall_fscore_support(_y_test, y_test_predict,
                                                                                  average='macro')
    test_accuracy = model.score(X_test_kbest, _y_test)

    return train_fscore, test_fscore, train_accuracy, test_accuracy


def plot_score_with_feature_selection(_features, model, _X_train_scaled, _y_train, _X_test_scaled, _y_test):
    train_fscores = []
    test_fscores = []
    k_list = range(5, len(_features) + 1)
    for k in k_list:
        print k
        train_fscore, test_fscore, train_accuracy, test_accuracy = run_with_feature_selection(model, k,  _X_train_scaled, _y_train, _X_test_scaled, _y_test)
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
    print('Train fscore: %.9f' % (train_fscore))

    # evaluation on the test set
    y_test_pred = gs.predict(_X_test)
    test_precision, test_recall, test_fscore, _ = precision_recall_fscore_support(_y_test, y_test_pred, average='macro')
    print('Test f-score: %.9f' % (test_fscore))