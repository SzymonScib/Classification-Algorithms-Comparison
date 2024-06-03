from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

df = pd.read_csv('./data/cancer_data.csv')

X = df.drop(['diagnosis', 'id', ], axis=1)
Y = df['diagnosis']

pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),  
    ('logregression', RandomForestClassifier())  
])

cv_predictions = cross_val_predict(pipeline, X, Y, cv=5)

conf_matrix = confusion_matrix(Y, cv_predictions, labels=['M', 'B'])
print('Confusion Matrix for all cross-validated predictions:')
print(conf_matrix)

class_report = classification_report(Y, cv_predictions, target_names=['M', 'B'], labels=['M', 'B'])
print('Classification Report for all cross-validated predictions:')
print(class_report)

cv_scores = cross_val_score(pipeline, X, Y, cv = 5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean():.2f}')
print(f'Standard deviation of cross-validation scores: {cv_scores.std():.2f}')