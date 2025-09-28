# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Function to plot confusion matrix
def plot_confusion_matrix(y, y_predict, model_name):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xticks([0.5,1.5], ['did not land', 'land'])
    plt.yticks([0.5,1.5], ['did not land', 'landed'])
    plt.show()

# -----------------------------
# Load datasets
# -----------------------------
URL1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
data = pd.read_csv(URL1)

URL2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv'
X = pd.read_csv(URL2)

# -----------------------------
# Prepare labels and features
# -----------------------------
Y = data['Class'].to_numpy()  # Labels as numpy array
transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)  # Standardize features

# Split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# -----------------------------
# Logistic Regression
# -----------------------------
parameters_lr = {"C":[0.01, 0.1, 1], 'penalty':['l2'], 'solver':['lbfgs']}
lr = LogisticRegression()
logreg_cv = GridSearchCV(lr, parameters_lr, cv=10)
logreg_cv.fit(X_train, Y_train)

lr_test_accuracy = logreg_cv.score(X_test, Y_test)
yhat_lr = logreg_cv.predict(X_test)
print("Logistic Regression - Best params:", logreg_cv.best_params_)
print("Logistic Regression - CV Accuracy:", logreg_cv.best_score_)
print("Logistic Regression - Test Accuracy:", lr_test_accuracy)
plot_confusion_matrix(Y_test, yhat_lr, "Logistic Regression")

# -----------------------------
# Support Vector Machine
# -----------------------------
parameters_svm = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                  'C': np.logspace(-3, 3, 5),
                  'gamma': np.logspace(-3, 3, 5)}
svm = SVC()
svm_cv = GridSearchCV(svm, parameters_svm, cv=10)
svm_cv.fit(X_train, Y_train)

svm_test_accuracy = svm_cv.score(X_test, Y_test)
yhat_svm = svm_cv.predict(X_test)
print("\nSVM - Best params:", svm_cv.best_params_)
print("SVM - CV Accuracy:", svm_cv.best_score_)
print("SVM - Test Accuracy:", svm_test_accuracy)
plot_confusion_matrix(Y_test, yhat_svm, "SVM")

# -----------------------------
# Decision Tree
# -----------------------------
parameters_tree = {'criterion': ['gini', 'entropy'],
                   'splitter': ['best', 'random'],
                   'max_depth': [2*n for n in range(1, 10)],
                   'max_features': ['auto', 'sqrt'],
                   'min_samples_leaf': [1, 2, 4],
                   'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree, parameters_tree, cv=10)
tree_cv.fit(X_train, Y_train)

tree_test_accuracy = tree_cv.score(X_test, Y_test)
yhat_tree = tree_cv.predict(X_test)
print("\nDecision Tree - Best params:", tree_cv.best_params_)
print("Decision Tree - CV Accuracy:", tree_cv.best_score_)
print("Decision Tree - Test Accuracy:", tree_test_accuracy)
plot_confusion_matrix(Y_test, yhat_tree, "Decision Tree")

# -----------------------------
# K-Nearest Neighbors
# -----------------------------
parameters_knn = {'n_neighbors': list(range(1, 11)),
                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'p': [1,2]}

KNN = KNeighborsClassifier()
knn_cv = GridSearchCV(KNN, parameters_knn, cv=10)
knn_cv.fit(X_train, Y_train)

knn_test_accuracy = knn_cv.score(X_test, Y_test)
yhat_knn = knn_cv.predict(X_test)
print("\nKNN - Best params:", knn_cv.best_params_)
print("KNN - CV Accuracy:", knn_cv.best_score_)
print("KNN - Test Accuracy:", knn_test_accuracy)
plot_confusion_matrix(Y_test, yhat_knn, "KNN")

# -----------------------------
# Compare all models
# -----------------------------
model_accuracies = {
    'Logistic Regression': lr_test_accuracy,
    'SVM': svm_test_accuracy,
    'Decision Tree': tree_test_accuracy,
    'KNN': knn_test_accuracy
}

best_model_name = max(model_accuracies, key=model_accuracies.get)
best_model_accuracy = model_accuracies[best_model_name]

print("\nBest performing model on test data:", best_model_name)
print("Test accuracy of best model:", best_model_accuracy)
