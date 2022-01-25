import pandas as pd
import numpy as np
import pickle
df=pd.read_excel('nature.xls')
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['saison']=encoder.fit_transform(df['saison'])
cleanup={"natureDuSinistre":{"M":1, "C": 0},"sexe":{"femme":1,"Homme":2}}  # at first, let's convert Promoted to numerical format
df.replace(cleanup, inplace=True)
# MIN MAX SCALING
from sklearn.preprocessing import MinMaxScaler
minmax_scale = MinMaxScaler().fit(df[['natureDuSinistre']])
df_minmax = minmax_scale.transform(df[['natureDuSinistre']])
#import relevant libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report
#features extraction

x = df[['CIN','age', 'sexe', 'jour', 'annee','saison']]
y = df['natureDuSinistre']
# Sampling minority data to match majority
# summarize class distribution
counter = Counter(y)
over = SMOTE(sampling_strategy=0.3) #oversampling minority to have 30% of majority
under = RandomUnderSampler(sampling_strategy=0.5)#undersampling the majority so it has only 50% more
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
# transform the dataset
x, y = pipeline.fit_resample(x, y)
# summarize class distribution
counter = Counter(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=0,stratify=y)  #splitting data with test size of 25%


RF=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                        criterion='gini', max_depth=None, max_features=6,
                        max_leaf_nodes=None, max_samples=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators=90,
                        n_jobs=None, oob_score=False, random_state=None,
                        verbose=0, warm_start=False)

RF.fit(x_train,y_train)
random_forest_training_score = 100*RF.score(x_train,y_train)
random_forest_test_score = 100*RF.score(x_test,y_test)
print("Random forest accuracy, Train : {:.2f}%, Test: {:.2f}%. ".format(random_forest_training_score, random_forest_test_score))
y_pred=RF.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
pickle.dump(RF,open('modell.pkl','wb'))


