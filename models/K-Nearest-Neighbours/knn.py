from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from math import sqrt
import pandas as pd
import numpy as np

df = pd.read_csv('./data/cancer_data.csv')

#Spliting the dataset into features "X" - the measurments used to determine wether a tumor is malignant/benign, and labels "Y" - the diagnosies 
X = df.drop(['diagnosis', 'id', ], axis=1)
Y = df['diagnosis']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

#Use the Square Root of N rule to pick an optimal k
k = int(sqrt(len(X_train)))

pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),  
    ('knn', KNeighborsClassifier(n_neighbors=k))  
])


pipeline.fit(X_train, Y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

conf_matrix = confusion_matrix(Y_test, y_pred, labels=['M', 'B'])
print('Confusion Matrix:')
print(conf_matrix)

class_report = classification_report(Y_test, y_pred, target_names=['M', 'B'], labels=['M', 'B'])
print('Classification Report:')
print(class_report)