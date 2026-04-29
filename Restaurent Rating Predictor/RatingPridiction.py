import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Dataset .csv")
print(df)
print(df.info())
print(df.describe())
print(df.duplicated().sum())
print(df.isna())
print(df.isna().sum())
print(df.columns)

print(df.groupby("City")["Average Cost for two"].mean())
print(df.groupby("City")["Average Cost for two"].mean().sort_values(ascending=False))
print(df.groupby("City")["Average Cost for two"].mean().sort_values(ascending=False).head())
print(df.groupby("City")["Average Cost for two"].mean().sort_values(ascending=False).head().plot(kind="bar"))
plt.title("Average cost for two per city, top 5")
plt.xlabel("City")
plt.ylabel("Amount")
plt.show()

print(df.groupby("Cuisines")["Votes"].sum().sort_values(ascending=True))
data = (df.groupby("Cuisines")["Votes"].sum().reset_index())
print(data)
print(data[data["Votes"]==0])
print(data[data["Votes"]!=0])

print(df.columns)
sns.pairplot(df[["Average Cost for two", "Aggregate rating", "Votes"]])
plt.show()

print(df["Has Online delivery"].value_counts())
print(df["Has Online delivery"].value_counts().plot(kind="pie"))
plt.table("Has Online delivery")
plt.ylabel("")
plt.legend()
plt.show()

print(df[["Aggregate rating", "Rating text"]].head())
print(df[["Aggregate rating", "Rating text"]].tail(10))
print(df[["Aggregate rating", "Rating text"]].sort_values(by="Aggregate rating", ascending = True))
print(df[["Aggregate rating", "Rating text"]].sort_values(by="Aggregate rating", ascending = True).head(5000))

df = (df[df["Rating text"] != "Not rated"])
print(df)
print(df["Rating text"].value_counts())
print(df["Rating text"].value_counts().plot(kind = "pie"))
plt.title("Rating Averages")
plt.ylabel("")
plt.show()

X = df[["Average Cost for two", "Has Table booking", "Has Online delivery", "Price range"]]

y = df[["Aggregate rating"]]
print(X)


from sklearn import preprocessing
labelencoder = preprocessing.LabelEncoder()

X["Has Table booking"] = labelencoder.fit_transform(X["Has Table booking"])
print(X)

print(labelencoder.classes_)

X["Has Online delivery"] = labelencoder.fit_transform(X["Has Online delivery"])
print(X)

print(labelencoder.classes_)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = scaler.fit_transform(X)
print(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.metrics import mean_absolute_error, mean_squared_error
def modelresults(predictions):
    print("Mean absolute error on model is {}".format(mean_absolute_error(y_test, predictions)))
    print("Root mean squared error on model is {}".format(np.sqrt(mean_squared_error(y_test, predictions))))


from sklearn.linear_model import LinearRegression
lr = LinearRegression()

print(lr.fit(X_train, y_train))
#print(lr.predict(X_test))
predictionsfromlr = (lr.predict(X_test))
modelresults(predictionsfromlr)
print(predictionsfromlr)

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
svrmodel = SVR()

param_gridsvr = {"C": [0.001, 0.01, 0.1, 0.5], "kernel" : ["linear", "rbf", "poly"], "degree" : [2, 3, 4]}
gridsvr = GridSearchCV(svrmodel, param_gridsvr)
gridsvr.fit(X_train, y_train)

print(gridsvr.predict(X_test))
predsgridsvr = gridsvr.predict(X_test)
print(modelresults(predsgridsvr))


from sklearn.tree import DecisionTreeRegressor
param_grid = {
    "max_depth" : [10, 30],
    "min_samples_leaf" : [1,2],
    "min_samples_split" : [2,5]

}

treemodel = DecisionTreeRegressor()
grid_tree = GridSearchCV(estimator= treemodel, param_grid= param_grid)

print(grid_tree.fit(X_train, y_train))
print(grid_tree.predict(X_test))
treepredictions = (grid_tree.predict(X_test))

print(modelresults(treepredictions))


from sklearn.ensemble import RandomForestRegressor
rfrmodel = RandomForestRegressor()

param_gridrfr = {"max_depth": [5,15], "n_estimators" : [2, 5, 10]}
gridrfr = GridSearchCV(rfrmodel, param_gridrfr)
print(gridrfr.fit(X_train, y_train))

print(gridrfr.predict(X_test))
randomforestpredictions = gridrfr.predict(X_test)
print(modelresults(randomforestpredictions))

print(gridrfr.best_params_)
print(grid_tree.best_params_)
print(gridrfr.best_params_)


from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
knn_param_grid = {"n_neighbors" : [9, 11 , 13, 14]}
knn_grid_search = GridSearchCV(knn, knn_param_grid)

print(knn_grid_search.fit(X_train, y_train))
knnpreds = knn_grid_search.predict(X_test)
print(modelresults(knnpreds))

print(knn_grid_search.best_params_)


from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor()
ada_param_grid = {"n_estimators" : [50, 100, 200],
                  "learning_rate" : [0.1, 0.5, 1]}
ada_grid_search = GridSearchCV(ada, ada_param_grid)

print(ada_grid_search.fit(X_train, y_train))
print(ada_grid_search.predict(X_test))
adapreds = ada_grid_search.predict(X_test)
print(modelresults(adapreds))
print(gridrfr)
