import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
sns.set()
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('mode.chained_assignment', None)

#!pip install xgboost
#!pip install tensorflow

from scipy.stats import skew
from sklearn import model_selection
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, precision_score, recall_score, plot_roc_curve 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from scipy import stats
from scipy.stats import zscore
from scipy.stats.mstats import winsorize 

from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from sklearn.preprocessing import RobustScaler

import plotly.graph_objects as go
import tensorflow as tf

import plotly.express as px

def get_data():
    data = pd.read_csv("data/heart_disease_dataset.csv")
    data1=data.dropna(axis=0)
    clean_data=data1.drop(["chol","fbs"],axis=1)
    return clean_data

df=get_data()


#detecting and cleaning outliers

#for resting blood pressure feature outliers
winsorize_percentile_trtbps = (stats.percentileofscore(df["trtbps"],165))/100
trtbps_winsorize = winsorize(df.trtbps,(0,(1-winsorize_percentile_trtbps)))
df["trtbps_winsorize"]= trtbps_winsorize


#for Thalach feature outliers
def iqr(df, var):
    q1 = np.quantile(df[var], 0.25)
    q3 = np.quantile(df[var], 0.75)
    diff = q3 - q1
    lower_v = q1 - (1.5 * diff)
    upper_v = q3 + (1.5 * diff)
    return df[(df[var] < lower_v) | (df[var] > upper_v)]

thalachh_out = iqr(df, "thalachh")
a=thalachh_out.index[0]
df= df.drop(a,axis=0)

#for Oldpeak
winsorize_percentile_oldpeak = (stats.percentileofscore(df["oldpeak"], 4)) / 100
oldpeak_winsorize = winsorize(df.oldpeak, (0, (1 - winsorize_percentile_oldpeak)))
df["oldpeak_winsorize"] = oldpeak_winsorize

#drop column that less correlate
df = df.drop(["trtbps", "oldpeak"], axis = 1)

#Applying One hot encoding method to categorical variables
numeric_var = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
categoric_var = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall', 'output']
categoric_var.remove("fbs")
df = pd.get_dummies(df, columns = categoric_var[:-1], drop_first = True)

#Feature scaling with the RobustScaler method of continuous and numerical variables for ML models use
final_dataset = df.copy()
new_numeric_var = ["age", "thalachh", "trtbps_winsorize", "oldpeak_winsorize"]
robust_scaler = RobustScaler()
final_dataset[new_numeric_var] = robust_scaler.fit_transform(df[new_numeric_var])


print(final_dataset)
