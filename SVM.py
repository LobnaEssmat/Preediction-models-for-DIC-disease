from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
from sklearn import svm as sv
import tensorflow as tf
from sklearn import preprocessing
#from tensorflow import feature_column
#from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn import datasets
#le = preprocessing.LabelEncoder()

"""
#dfs = pd.read_excel("C:\\Users\\hp\\Downloads\\GP\\S2 File.xlsx", sheet_name=None)
xl_file = pd.ExcelFile("C:\\Users\\hp\\Downloads\\GP\\S2 File.xlsx")
#xl_file=pd.read_excel("C:\\Users\\hp\\Downloads\\GP\\S2 File.xlsx")
set1=[]
set2=[]
dfs = {sheet_name: xl_file.parse(sheet_name)
          for sheet_name in xl_file.sheet_names}
target=pd.ExcelFile("C:\\Users\\hp\\Downloads\\GP\\target.xlsx")

trgt={sheet_name: target.parse(sheet_name)
          for sheet_name in target.sheet_names}
"""

col_names = ['Sex','Age','ICU','APACHE2','expire','DIC','Anticoagulation','Thrombosis','Bleeding','organ failure','SIRS','Infection','Tissue',
             'Surgery','Hema','Cancer','Liver','Obstetric','Vascular', 'Immunologic','organ transplant','ISTH','PT',
             'PT(%)','INR','aPTT','Fibrinogen','D-Dimer','TT','AT','III','Protein C','WBC', 'RBC', 'Hb', 'Hct', 'MCV',
             'MCH', 'MCHC', 'RDW', 'PDW(fL)', 'MPV', 'PLT' ,'Neu(%)', 'Lympho(%)','Mono(%)', 'Eo(%)', 'Baso(%)', 'Rate']

# load dataset
pima = pd.read_csv("C:\\Users\\hp\\Downloads\\S2_File (1).csv", header=None, names=col_names)
data=pima.replace('-', np.NaN)
preprocessed_data=data.fillna(0)

#preprocessed_data= le.fit_transform(preprocessed_data.astype(str))
#split dataset in features and target variable
feature_cols = ['Sex','Age','ICU','APACHE2','expire','Anticoagulation','Thrombosis','Bleeding','organ failure','SIRS','Infection','Tissue',
             'Surgery','Hema','Cancer','Liver','Obstetric','Vascular', 'Immunologic','organ transplant','ISTH','PT',
             'PT(%)','INR','aPTT','Fibrinogen','D-Dimer','TT','AT','III','Protein C','WBC', 'RBC', 'Hb', 'Hct', 'MCV',
             'MCH', 'MCHC', 'RDW', 'PDW(fL)', 'MPV', 'PLT' ,'Neu(%)', 'Lympho(%)','Mono(%)', 'Eo(%)', 'Baso(%)', 'Rate']
X = preprocessed_data[feature_cols] # Features
y = preprocessed_data.DIC # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=9) # 70% training and 30% test




#Create a svm Classifier
clf = sv.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
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

##############################################################
"""
#visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(X_train)
pca_2d = pca.transform(X_train)
svmClassifier_2d =   sv.LinearSVC(random_state=111).fit(pca_2d, y_train)
import matplotlib.pyplot as pl
c1=''
c2=''
for i in range(0, pca_2d.shape[0]):
 if y_train[i] == 0:
  c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',    marker='+')
 elif y_train[i] == 1:
  c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',    marker='o')

 pl.legend([c1, c2], ['non DIC', 'DIC'])
 x_min, x_max = pca_2d[:, 0].min() - 1, pca_2d[:, 0].max() + 1
 y_min, y_max = pca_2d[:, 1].min() - 1, pca_2d[:, 1].max() + 1
 xx, yy = np.meshgrid(np.arange(x_min, x_max, .01), np.arange(y_min, y_max, .01))
 Z = svmClassifier_2d.predict(np.c_[xx.ravel(), yy.ravel()])
 Z = Z.reshape(xx.shape)
 pl.contour(xx, yy, Z)
 pl.title('Support Vector Machine Decision Surface')
 pl.axis('off')
 pl.show()
"""


