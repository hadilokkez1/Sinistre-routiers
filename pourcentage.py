import numpy as np
import pandas as pd
import pickle
import sklearn
dataset = pd.read_excel('pourf.xls')
#dropping usuless coloumns

dataset["age"].fillna( method ='ffill', inplace = True)
dataset['age'] = dataset['age'].astype(int)
#encodage
cleanup={"natureDuSinistre":{"M":1, "C": 0},"sexe":{"femme":1,"Homme":2}}  # at first, let's convert natureDuSinistre to numerical format
dataset.replace(cleanup, inplace=True)
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
dataset['saison']=encoder.fit_transform(dataset['saison'])
dataset['nomPrenomRaisonSocial']=encoder.fit_transform(dataset['nomPrenomRaisonSocial'])
# Take a better look at categorical data
cat_columns = dataset.select_dtypes(include = ['object'])
unique_values = cat_columns.nunique(dropna=False)
print (unique_values)
# Take a better look at categorical data
cat_columns = dataset.select_dtypes(include = ['object'])
unique_values = cat_columns.nunique(dropna=False)
print (unique_values)
x = dataset[['CIN','age', 'sexe', 'natureDuSinistre', 'jour', 'mois', 'annee','saison']]
y = dataset['pourcentadeDeResponsabilite']
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.metrics import confusion_matrix,accuracy_score

over = RandomOverSampler(sampling_strategy={0:9913,25:5000,50:5000,75:5000,100:14067})
under = RandomUnderSampler(sampling_strategy={0:5000,100:5000})
steps = [('o', over),('u', under)]
pipeline = Pipeline(steps=steps)
counter = Counter(y)

# transform the dataset
#x, y = pipeline.fit_resample(x, y) uncomment this if u need more accuracy
x,y= over.fit_resample(x, y) #less accuracy but all original data remains
# summarize the new class distribution
counter = Counter(y)

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.35, random_state = 0,stratify=y)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
from sklearn.metrics import classification_report
y_pred  =  clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)

print(classification_report(y_test, y_pred))
#Improving randomforest
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 15, num = 2)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = random_grid, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
pickle.dump(clf,open('modelll.pkl','wb'))