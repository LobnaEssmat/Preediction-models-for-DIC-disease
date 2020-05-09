# Load libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
from  sklearn import  tree
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

#le = preprocessing.LabelEncoder()
col_names = ['Sex','Age','ICU','APACHE2','expire','DIC','Anticoagulation','Thrombosis','Bleeding','organ failure','SIRS','Infection','Tissue',
             'Surgery','Hema','Cancer','Liver','Obstetric','Vascular', 'Immunologic','organ transplant','ISTH','PT',
             'PT(%)','INR','aPTT','Fibrinogen','D-Dimer','TT','AT','III','Protein C','WBC', 'RBC', 'Hb', 'Hct', 'MCV',
             'MCH', 'MCHC', 'RDW', 'PDW(fL)', 'MPV', 'PLT' ,'Neu(%)', 'Lympho(%)','Mono(%)', 'Eo(%)', 'Baso(%)', 'Rate']

# load dataset
pima = pd.read_csv("C:\\Users\\hp\\Downloads\\S2_File (1).csv", header=None, names=col_names)
data=pima.replace('-', np.NaN)
preprocessed_data=data.fillna(0)
#preprocessed_datae= le.fit_transform(preprocessed_data.astype(str))

#split dataset in features and target variable
feature_cols = ['Sex','Age','ICU','APACHE2','expire','Anticoagulation','Thrombosis','Bleeding','organ failure','SIRS','Infection','Tissue',
             'Surgery','Hema','Cancer','Liver','Obstetric','Vascular', 'Immunologic','organ transplant','ISTH','PT',
             'PT(%)','INR','aPTT','Fibrinogen','D-Dimer','TT','AT','III','Protein C','WBC', 'RBC', 'Hb', 'Hct', 'MCV',
             'MCH', 'MCHC', 'RDW', 'PDW(fL)', 'MPV', 'PLT' ,'Neu(%)', 'Lympho(%)','Mono(%)', 'Eo(%)', 'Baso(%)', 'Rate']



X = preprocessed_data[feature_cols] # Features
y = preprocessed_data.DIC # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=9) # 70% training and 30% test
"""
#condition for vte ---> 2 is label for vte
if ( (preprocessed_data.Age[84] > 40) or (preprocessed_data.Surgery[84] ==1) or (preprocessed_data.Cancer[84] ==1) or
        (preprocessed_data.Vascular[84]==1) ):
    preprocessed_data.DIC[84]=2
"""
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# calculate the percentage of zeros
y_test.mean()
# calculate null accuracy in a single line of code
# only for binary classification problems coded as 0/1
max(y_test.mean(), 1 - y_test.mean())
# calculate null accuracy (for multi-class classification problems)
y_test.value_counts().head(1) / len(y_test)
confusion = metrics.confusion_matrix(y_test, y_pred)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
sensitivity = TP / float(FN + TP)

print("sensitivity : ",sensitivity)
print(metrics.recall_score(y_test, y_pred))
specificity = TN / (TN + FP)

print("specificity :",specificity)
################################

#density plotting
import seaborn as sns
sns.set()
#plot=sns.distplot(preprocessed_data['DIC'],hist=False, bins=10)
dat=preprocessed_data['DIC']
plt.hist(dat,bins=10)
plt.show()
##################################

confusion_matrix = metrics.confusion_matrix(y_test, y_pred, labels=[1, 0])
print(confusion_matrix)




def drawConfusionMatrix( confusion_matrix):
    labels = ['DIC', 'non DIC']
    fig = plt.figure()

    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrix)
    fig.colorbar(cax)

    plt.title('Confusion matrix')
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
drawConfusionMatrix(confusion_matrix)

####################################################################
""""
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())
"""
####################################################################################

#optimize id3 and its depth

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

########################################################
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
# calculate null accuracy in a single line of code
# only for binary classification problems coded as 0/1
max(y_test.mean(), 1 - y_test.mean())
# calculate null accuracy (for multi-class classification problems)
y_test.value_counts().head(1) / len(y_test)
confusion = metrics.confusion_matrix(y_test, y_pred)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
sensitivity = TP / float(FN + TP)

print("sensitivity : ",sensitivity)
print(metrics.recall_score(y_test, y_pred))
specificity = TN / (TN + FP)

print("specificity :",specificity)
# f1 score
score = f1_score(y_pred, y_test)

# print
print ("Decision Tree F1 score: {:.2f}".format(score))

plt.plot(sensitivity, specificity, linestyle='--', label='sensitvity & specifity')
plt.xlabel('sensitivity')
plt.ylabel('specificity')
plt.legend()
plt.show()
tree.plot_tree(clf.fit(X_train, y_train))
plt.show()
##################################################################################

""""
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#visualize after optimization

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())
"""

