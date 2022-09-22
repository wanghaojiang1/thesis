import numpy as np
from . import match_service
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

def evaluate():
    edges = match_service._get_collective_labelled_edges()
    
    x = list(map(lambda edge: {**edge['scores']}, edges))
    y =  list(map(lambda edge: {'truth': 1 if edge['correct'] else 0 }, edges))

    X = pd.DataFrame(x)
    Y = pd.DataFrame(y)

    x = []
    train_y = []
    test_y = []
    deviation_y = []

    for i in range(1, 101):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2)
        smote = SMOTE()
        # fit predictor and target variable
        x_smote, y_smote = smote.fit_resample(X_train, y_train)

        classifier = SVC(kernel='rbf')
        classifier.fit(x_smote, y_smote)
        train_scores = cross_val_score(classifier, x_smote, y_smote.values.ravel())
        train_y.append(train_scores.mean())
        deviation_y.append(train_scores.std())

        test_score = classifier.score(X_test, y_test.values.ravel())
        test_y.append(test_score)
        # print(test_score)
        y_pred = classifier.predict(X_test)
        
        TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

        print(" ------------------- RUN " + str(i) + " ------------------- ")
        print("Accuracy: ", test_score)
        print("True positives: ", TP)
        print("False positives: ", FP)
        print("True negatives: ", TN)
        print("False negatives: ", FN)
        print(y_test['truth'].value_counts())
        x.append(i)
        plot_confusion_matrix(classifier, X_test, y_test)  
        plt.savefig('exports/confusion_matrix.png')
        plt.clf()
    

    accuracy = np.array(test_y)
    print('MEAN ACCURACY: ', np.mean(accuracy))
    print('STD ACCURACY:', np.std(accuracy))

    plt.plot(x, train_y, label='train_error')
    plt.fill_between(x, np.array(train_y) - np.array(deviation_y), np.array(train_y) + np.array(deviation_y),
                 color='gray', alpha=0.2)
    plt.plot(x, test_y, label='test_error')
    plt.ylabel('Score')
    plt.xlabel('Rounds')
    plt.legend(loc="upper right")
    plt.savefig('exports/SVM.png')
    plt.clf()

def evaluate_learning_curve():
    edges = match_service._get_collective_labelled_edges()
    x = list(map(lambda edge: {**edge['scores']}, edges))
    y =  list(map(lambda edge: {'truth': 1 if edge['correct'] else 0 }, edges))

    X = pd.DataFrame(x)
    Y = pd.DataFrame(y)

    classifier = SVC(kernel='rbf')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2)
    train_sizes, train_scores, test_scores = learning_curve(estimator=classifier, X=X_train, y=y_train,
                                                       cv=10, train_sizes=np.linspace(0.1, 1.0, 10),
                                                     n_jobs=1)
    #
    # Calculate training and test mean and std
    #
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    #
    # Plot the learning curve
    #
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.title('Learning Curve')
    plt.xlabel('Training Data Size')
    plt.ylabel('Model accuracy')
    plt.grid()
    plt.legend(loc='lower right')
    plt.savefig('exports/learning_curve.png')
    plt.show()


def evaluate_messy():
    edges = match_service._get_collective_labelled_edges()
    x = list(map(lambda edge: {**edge['scores']}, edges))
    y =  list(map(lambda edge: {'truth': 1 if edge['correct'] else 0 }, edges))

    X = pd.DataFrame(x)
    Y = pd.DataFrame(y)

    # Train classifier
    # classifier = SVC(kernel='rbf')
    # classifier.fit(X_train, y_train)

    train_errors = []
    test_errors = []

    # for i in range (0, 100):

    #     train_score = classifier.score(X_train, y_train)
    #     test_score = classifier.score(X_test, y_test)

    #     train_errors.append(train_score)
    #     test_errors.append(test_score)
    x = []
    train_y = []
    test_y = []
    deviation_y = []
    for i in range(1, 101):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2)
        print(y_train['truth'].value_counts())
        print(y_test['truth'].value_counts())
        classifier = SVC(kernel='rbf')
        classifier.fit(X_train, y_train)
        train_scores = cross_val_score(classifier, X_train, y_train.values.ravel())
        train_y.append(train_scores.mean())
        deviation_y.append(train_scores.std())

        # print(y_train)
        # test_scores = cross_val_score(classifier, X_test, y_test.values.ravel())
        # print("TESTING", test_scores)
        test_score = classifier.score(X_test, y_test.values.ravel())
        test_y.append(test_score)
        
        # predicted = classifier.predict(X_test)

        # # True positives
        # TP = y_test[y_test & predicted]

        # # False positives
        # FP = y_test[(y_test == 0) & predicted]

        # # True negatives
        # TN = y_test[(y_test == 0) & (predicted == 0)]

        # # False negatives
        # FN = y_test[y_test & (predicted == 0)]

        x.append(i)
        # print(len(TP), len(FP), len(TN), len(FN))
    
    # print(train_y)
    # print(deviation_y)
    # print(test_y)
    plt.plot(x, train_y, label='train_error')
    plt.fill_between(x, np.array(train_y) - np.array(deviation_y), np.array(train_y) + np.array(deviation_y),
                 color='gray', alpha=0.2)
    plt.plot(x, test_y, label='test_error')
    plt.ylabel('Score')
    plt.xlabel('Rounds')
    plt.legend(loc="upper right")
    plt.savefig('exports/SVM.png')
    plt.clf()

    # cm = confusion_matrix(y_test,y_pred)
    # accuracy = float(cm.diagonal().sum())/len(y_test)
    # print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)


def evaluate_clean():
    edges = match_service._get_collective_labelled_edges()
    x = list(map(lambda edge: {**edge['scores']}, edges))
    y =  list(map(lambda edge: {'truth': 1 if edge['correct'] else 0 }, edges))

    X = pd.DataFrame(x)
    Y = pd.DataFrame(y)

    x = []
    train_y = []
    test_y = []
    deviation_y = []
    for i in range(1, 101):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2)

        classifier = SVC(kernel='rbf')
        classifier.fit(X_train, y_train)
        train_scores = cross_val_score(classifier, X_train, y_train.values.ravel())
        train_y.append(train_scores.mean())
        deviation_y.append(train_scores.std())

        test_score = classifier.score(X_test, y_test.values.ravel())
        test_y.append(test_score)
        

        x.append(i)
    
    plt.plot(x, train_y, label='train_error')
    plt.fill_between(x, np.array(train_y) - np.array(deviation_y), np.array(train_y) + np.array(deviation_y),
                 color='gray', alpha=0.2)
    plt.plot(x, test_y, label='test_error')
    plt.ylabel('Score')
    plt.xlabel('Rounds')
    plt.legend(loc="upper right")
    plt.savefig('exports/SVM.png')
    plt.clf()