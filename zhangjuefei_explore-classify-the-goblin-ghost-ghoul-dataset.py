# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.decomposition import PCA

from sklearn.feature_selection import chi2, SelectFromModel

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.grid_search import GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

numeric_columns = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']

# data.groupby('type')[numeric_columns].describe()
fig = plt.figure(figsize=(8, 8))

idx = 1

for f in numeric_columns:

    ax = fig.add_subplot(2, len(numeric_columns) / 2, idx)

    idx += 1

    _ = sns.boxplot(x='type', y=f, data=data, palette='muted', ax=ax)
_ = sns.pairplot(data, vars=numeric_columns, hue='type', palette='muted')
color_one_hot = data.color.str.get_dummies()

data = pd.concat([data, color_one_hot], axis=1)

feature_columns = np.hstack([numeric_columns, color_one_hot.columns.values])

# data.head(10)
_, chi2_p_values = chi2(data[feature_columns], data.type)

print('chi2 p-values of features:')

for f, p in zip(feature_columns, chi2_p_values):

    print('{}:\t{:.5f}'.format(f, p))
dtc = DecisionTreeClassifier()

select_from_model = SelectFromModel(dtc)

select_from_model.fit(data[feature_columns], data.type)



print('\nfeatures chosen by feature importances of decision tree model:')

for f, c in zip(data[feature_columns], select_from_model.get_support()):

    print('{}:\t{}'.format(f, 'yes' if c else 'no'))
lrc = LogisticRegression(multi_class='multinomial', solver='lbfgs')

select_from_model = SelectFromModel(lrc)

select_from_model.fit(data[feature_columns], data.type)



print('\nfeatures chosen by feature importances of LR model:')

for f, c in zip(data[feature_columns], select_from_model.get_support()):

    print('{}:\t{}'.format(f, 'yes' if c else 'no'))
pca = PCA()

pcas = pca.fit_transform(data[['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']])

pcas = pd.DataFrame(pcas, columns=['pc{:d}'.format(i) for i in np.arange(1, 5)])

pcas['type'] = data.type



# pair scatter plot for 4 princple components.

_ = sns.pairplot(pcas, vars=['pc1', 'pc2', 'pc3', 'pc4'], hue='type', palette='muted')
# labels for ghoul v.s. non-ghoul.

data['is_ghoul'] = (data.type == 'Ghoul').astype(np.int)



# parameter grid

params = {

    'C': [50, 100, 200],

    'kernel': ['poly'],

    'degree': [2, 4, 6],

    # 'gamma': [1, 5, 10],

    # 'coef0': [1, 5, 10]

}



# SVM for separate ghoul from others.

ghoul_svc = SVC(probability=True)



# split the train and test set.

train_features, test_features, train_labels, test_labels = train_test_split(data[feature_columns], data.is_ghoul,

                                                                            train_size=0.8)

# grid search.

gs = GridSearchCV(estimator=ghoul_svc, param_grid=params, cv=3, refit=True, scoring='accuracy')

gs.fit(train_features, train_labels)

ghoul_svc = gs.best_estimator_



print('\nBest parameters:')

for param_name, param_value in gs.best_params_.items():

    print('{}:\t{}'.format(param_name, str(param_value)))



print('\nBest score (accuracy): {:.3f}'.format(gs.best_score_))



# merics.

predict_labels = gs.predict(test_features)

predict_proba = gs.predict_proba(test_features)

fpr, rc, th = roc_curve(test_labels, predict_proba[:, 1])

precision, recall, threshold = precision_recall_curve(test_labels, predict_proba[:, 1])

roc_auc = auc(fpr, rc)



print('\nMetrics: Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, AUC: {:.3f}'.format(accuracy_score(test_labels, predict_labels), precision_score(test_labels, predict_labels), recall_score(test_labels, predict_labels), roc_auc))

print('\nClassification Report:')

print(classification_report(test_labels, predict_labels, target_names=['non-ghoul', 'ghoul']))



# draw some charts.

fig = plt.figure(figsize=(16, 4))

ax = fig.add_subplot(131)

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('Recall')

ax.set_title('ROC Curve')

ax.plot(fpr, rc, 'b')

ax.plot([0.0, 1.0], [0.0, 1.0], 'r--')

ax.text(0.80, 0.05, 'auc: {:.2f}'.format(roc_auc))



ax = fig.add_subplot(132)

ax.set_xlabel('Threshold')

ax.set_ylabel('Precision & Recall')

ax.set_title('Precsion & Recall')

ax.set_xlim([threshold.min(), threshold.max()])

ax.set_ylim([0.0, 1.0])

ax.plot(threshold, precision[:-1], 'b', label='Precision')

ax.plot(threshold, recall[:-1], 'r', label='Recall')

_ = ax.legend(loc='best')



ts = np.arange(0, 1.02, 0.02)

accuracy = []

for t in ts:

    predict_label = (predict_proba[:, 1] >= t).astype(np.int)

    accuracy.append(accuracy_score(test_labels, predict_label))



ax = fig.add_subplot(133)

ax.set_xlabel("Threshold")

ax.set_ylabel("Accuracy")

ax.set_ylim([0.0, 1.0])

ax.set_title('Accuracy')

ax.plot([0.0, 1.0], [0.5, 0.5], 'r--')

ax.plot(ts, accuracy, 'b')



plt.show()
# labels for ghost v.s. non-ghost.

data['is_ghost'] = (data.type == 'Ghost').astype(np.int)



# SVM for separate ghost from others.

ghost_svc = SVC(probability=True)



# split the train and test set.

train_features, test_features, train_labels, test_labels = train_test_split(data[feature_columns], data.is_ghost,

                                                                            train_size=0.8)

# grid search.

gs = GridSearchCV(estimator=ghost_svc, param_grid=params, cv=3, refit=True, scoring='accuracy')

gs.fit(train_features, train_labels)

ghost_svc = gs.best_estimator_



print('\nBest parameters:')

for param_name, param_value in gs.best_params_.items():

    print('{}:\t{}'.format(param_name, str(param_value)))



print('\nBest score (accuracy): {:.3f}'.format(gs.best_score_))



# merics.

predict_labels = gs.predict(test_features)

predict_proba = gs.predict_proba(test_features)

fpr, rc, th = roc_curve(test_labels, predict_proba[:, 1])

precision, recall, threshold = precision_recall_curve(test_labels, predict_proba[:, 1])

roc_auc = auc(fpr, rc)



print('\nMetrics: Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, AUC: {:.3f}'.format(accuracy_score(test_labels, predict_labels), precision_score(test_labels, predict_labels), recall_score(test_labels, predict_labels), roc_auc))

print('\nClassification Report:')

print(classification_report(test_labels, predict_labels, target_names=['non-ghost', 'ghost']))



# draw some charts.

fig = plt.figure(figsize=(16, 4))

ax = fig.add_subplot(131)

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('Recall')

ax.set_title('ROC Curve')

ax.plot(fpr, rc, 'b')

ax.plot([0.0, 1.0], [0.0, 1.0], 'r--')

ax.text(0.80, 0.05, 'auc: {:.2f}'.format(roc_auc))



ax = fig.add_subplot(132)

ax.set_xlabel('Threshold')

ax.set_ylabel('Precision & Recall')

ax.set_title('Precsion & Recall')

ax.set_xlim([threshold.min(), threshold.max()])

ax.set_ylim([0.0, 1.0])

ax.plot(threshold, precision[:-1], 'b', label='Precision')

ax.plot(threshold, recall[:-1], 'r', label='Recall')

_ = ax.legend(loc='best')



ts = np.arange(0, 1.02, 0.02)

accuracy = []

for t in ts:

    predict_label = (predict_proba[:, 1] >= t).astype(np.int)

    accuracy.append(accuracy_score(test_labels, predict_label))



ax = fig.add_subplot(133)

ax.set_xlabel("Threshold")

ax.set_ylabel("Accuracy")

ax.set_ylim([0.0, 1.0])

ax.set_title('Accuracy')

ax.plot([0.0, 1.0], [0.5, 0.5], 'r--')

ax.plot(ts, accuracy, 'b')



plt.show()
train_features, test_features, train_labels, test_labels = train_test_split(data[feature_columns], data.type,

                                                                            train_size=0.8)



gbc = RandomForestClassifier()

params = {

    'criterion': ['gini', 'entropy'],

    'n_estimators': [50, 100, 200],

    'max_depth': [3, 5, 10]

}



gs = GridSearchCV(estimator=gbc, param_grid=params, cv=3, refit=True, scoring='accuracy')

gs.fit(train_features, train_labels)



print('\nBest Parameters:')

for param_name, param_value in gs.best_params_.items():

    print('{}:\t{}'.format(param_name, str(param_value)))



print('\nBest Score({}): {:.3f}'.format('Accuracy', gs.best_score_))



train_predict_labels = gs.predict(train_features)

print('\nTrain Classification Report:')

print(classification_report(train_labels, train_predict_labels))



predict_labels = gs.predict(test_features[feature_columns])

print('\nTest Classification Report:')

print(classification_report(test_labels, predict_labels))



cm = confusion_matrix(test_labels, predict_labels)

# cm = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]



_ = sns.heatmap(cm, square = True, xticklabels = ["ghost", "ghoul", "goblin"], annot = True, annot_kws = {"fontsize": 13}, yticklabels = ["ghost", "ghoul", "goblin"], cbar = True, cbar_kws = {"orientation": "horizontal"}, cmap = "Blues").set(xlabel = "predicted type", ylabel = "true type", title = "Confusion Matrix")
predict_as_ghoul = ghoul_svc.predict(test_features).astype(np.bool)

predict_as_ghost = ghost_svc.predict(test_features).astype(np.bool)



test_labels_pair = pd.DataFrame({'true_type': test_labels})

test_labels_pair['predict_type'] = 'Goblin'

test_labels_pair.loc[predict_as_ghoul, 'predict_type'] = 'Ghoul'

test_labels_pair.loc[predict_as_ghost, 'predict_type'] = 'Ghost'



cm = confusion_matrix(test_labels_pair.true_type, test_labels_pair.predict_type)

print('Accuracy: {:.3f}'.format(cm.diagonal().sum().astype(np.float32) / cm.sum().astype(np.float32)))

print('\nClassification Report:')

print(classification_report(test_labels_pair.true_type, test_labels_pair.predict_type))



# change to percentage.

# cm = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]

_ = sns.heatmap(cm, square = True, xticklabels = ["ghost", "ghoul", "goblin"], annot = True, annot_kws = {"fontsize": 13}, yticklabels = ["ghost", "ghoul", "goblin"], cbar = True, cbar_kws = {"orientation": "horizontal"}, cmap = "Blues").set(xlabel = "predicted type", ylabel = "true type", title = "Confusion Matrix")
# A two hidden layer network.

nw = MLPClassifier(hidden_layer_sizes = (30, 30), activation='logistic', alpha=0.001, solver='lbfgs', learning_rate='constant')

                   

nw.fit(train_features, train_labels)



train_predict_labels = nw.predict(train_features)

print('\nTrain Classification Report:')

print(classification_report(train_labels, train_predict_labels))



predict_labels = nw.predict(test_features)

print('\nTest Classification Report:')

print(classification_report(test_labels, predict_labels))



cm = confusion_matrix(test_labels, predict_labels)

# cm = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]



_ = sns.heatmap(cm, square = True, xticklabels = ["ghost", "ghoul", "goblin"], annot = True, annot_kws = {"fontsize": 13}, yticklabels = ["ghost", "ghoul", "goblin"], cbar = True, cbar_kws = {"orientation": "horizontal"}, cmap = "Blues").set(xlabel = "predicted type", ylabel = "true type", title = "Confusion Matrix")