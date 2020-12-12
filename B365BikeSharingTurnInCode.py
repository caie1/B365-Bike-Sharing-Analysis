import numpy as np    
import pandas as pd     
import matplotlib.pyplot as plt     
import seaborn as sns     
from sklearn.preprocessing import StandardScaler   # Standardize numirical data
from sklearn.preprocessing import PolynomialFeatures     # construct polynomial X
from sklearn.pipeline import make_pipeline        # combine PolynomialFeatures and linear regression
from sklearn.model_selection import GridSearchCV   # find best hypo paremeter
from sklearn.model_selection import KFold       # separate data to k fold
from sklearn.model_selection import cross_val_score   # preform cross validation to dataset
from sklearn.linear_model import Ridge       
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
df = pd.read_csv("SeoulBikeData.csv",encoding= 'unicode_escape')
df.head()
df.dtypes
df.isnull().sum()   # no null data
df.groupby('Seasons').first()
df = df.drop(["Date"],axis=1)      # Date column is basicly useless, let's drop it.
# one-hot encoding: categorical variables change to varoables like 001 010 100.
df = pd.get_dummies(df, drop_first=True)
print(df.corr()['Rented Bike Count'].sort_values())    
# The dataset gets more sense to us when the correlation with Rented Bike Count get's clear.
# Two typical relation graph
plt.title('Relation between Hour and Rented Bike Count')
sns.scatterplot(x=df['Hour'],y=df['Rented Bike Count'])
plt.show()
plt.title('Relation between Temperature and Rented Bike Count')
sns.scatterplot(x=df['Temperature(°C)'],y=df['Rented Bike Count'])
plt.show()
# prepare X and y
X = df.drop(['Rented Bike Count'], axis = 1)
y = df['Rented Bike Count']
# standardize X
sc = StandardScaler()
X = sc.fit_transform(X)
DecTr_param = {
    "min_samples_split": [10, 15],        #15
    "max_depth": [12, 14],                #14
    "min_samples_leaf": [10],             #10
    "max_leaf_nodes": [180, 210, 250],    #250
}
DecTr_grid = GridSearchCV(DecisionTreeRegressor(), DecTr_param, cv=5)

DecTr_grid.fit(X,y)
best_DT = DecTr_grid.best_estimator_
best_DT # The best parameters
RanForest_param = {
    'max_depth': [12,14],        #14
    'min_samples_leaf': [2],     #2
    'min_samples_split': [2],    #2
    'n_estimators': [100,120]    #100
}
RF_grid = GridSearchCV(RandomForestRegressor(), RanForest_param, cv=5)

RF_grid.fit(X,y.ravel())
best_RF = RF_grid.best_estimator_
best_RF
def mylinearRegression(X,y):
    # theta = (inv(X.T*X))*X.T*y
    dim = X.shape
    b = np.ones((dim[0],1))
    X = np.concatenate([X,b],axis=1)
    a = np.linalg.inv(np.dot(X.T,X))
    b = np.dot(X.T,y)
    theta = np.dot(a,b)
    return theta

def lrpredict(X,theta):
    # y_hat = X*theta
    dim = X.shape
    b=np.ones((dim[0],1))
    X = np.concatenate([X,b],axis=1)
    return np.dot(X,theta)

# metrics mean square error, haven't been used in this project
def lrmse(yhat,y):
    # mse=||y_hat-y||^2 /n 
    num = len(y)
    error = np.sum((yhat-y)**2)/num 
    return error

# metrics R_square: 
# 1 - residual sum of square / total sum of squares
def lrr2(yhat,y):
    sse = np.sum((yhat - y)**2)
    sst = np.sum((y - y.mean())**2)
    r_square = 1 - (sse/sst)
    return r_square
kf = KFold(n_splits = 5, random_state=1, shuffle=True)
my_lr_r2score = []    #used to store 5 r2 scores
for train_idx, test_idx in kf.split(X, y):     # Every loop will generate one model and one r2 score
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    theta = mylinearRegression(X_train,y_train)
    yhat = lrpredict(X_test,theta)
    r2 = lrr2(yhat,y_test)
    my_lr_r2score.append(r2)
r2 = sum(my_lr_r2score)/len(my_lr_r2score)  # mean of five r2 score
print("Our self implemented linear regression with 5 fold cross validation r2 score is: \n", r2)
polyreg=make_pipeline(PolynomialFeatures(degree=2),LinearRegression())
lreg = LinearRegression()
ridge = Ridge(alpha=0.5)
KNN = KNeighborsRegressor()

lr = cross_val_score(lreg, X, y,scoring='r2', cv=kf)
lr = lr.mean()       # linear regression  r2:0.55   

polyr = cross_val_score(polyreg, X, y,scoring='r2', cv=kf)
polyr = polyr.mean()     # polynomial regression  r2:0.70

rig = cross_val_score(ridge, X, y, scoring='r2', cv=kf)
rig = rig.mean()        # ridge regression    r2:0.55

dt = cross_val_score(best_DT, X, y,scoring='r2', cv=kf)
dt = dt.mean()             # decision tree regression   r2:0.82

rf = cross_val_score(best_RF, X, y.ravel(),scoring='r2', cv=kf)
rf = rf.mean()          # Random Forest regression    r2:0.87 (best)

knn = cross_val_score(KNN, X, y,scoring='r2', cv=kf)
knn = knn.mean()     # K nearest neighbor      r2:0.79
methods = ['myLinR','LinReg','PolyReg','Ridge','DecisionTr','RanForest', 'KNN']
scores=np.array([r2,lr,polyr,rig,dt,rf,knn])
ind = [x for x,_ in enumerate(methods)]
plt.bar(ind, scores)
plt.xticks(ind, methods)
plt.xlabel("ML Models")
plt.ylabel("R2 score")
plt.ylim(0.4,0.9)
plt.title("The R2 score for different models")

print("The r2 score for our self implemented linear regression is: \n", r2)
print('\nThe r2 score for sklearn LinearRegression is the same:\n', lr)
print('\nThe r2 score for PloynomialRegression is:\n', polyr)
print('\nThe r2 score for Ridge Regression is:\n', rig)
print('\nThe r2 score for DecisionTree is:\n', dt)
print('\nThe r2 score for RandomForest is:\n', rf)
print('\nThe r2 score for KNN is:\n', knn)
df.describe()
