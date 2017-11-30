import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt

#Read in data
data = pd.read_excel('CleanInProgress.xlsx')
data.to_csv('DataClean.csv')

#Extract training data to fit for Containment Pressure 2
train_test = data[data['ContainmentPressure2'].notnull()]
train = train_test.iloc[:64]
test = train_test.iloc[65:]

label_train = train.iloc[:, 4].as_matrix()
feature_train = train.drop('ContainmentPressure2', 1).as_matrix()
label_test = test.iloc[:, 4].as_matrix()
feature_test = test.drop('ContainmentPressure2', 1).as_matrix()

#Use Imputer to replace NaN's
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp1 = imp.fit(feature_train)
imp2 = imp.fit(feature_test)
feature_train = imp1.transform(feature_train)
feature_test = imp2.transform(feature_test)


#Train regression tree
clf = DecisionTreeRegressor(max_depth=5)
clf.fit(feature_train, label_train)

#Fit tests data
label_predict = clf.predict(feature_test)

#Plot results
plt.figure()
plt.scatter(test.index, label_test, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(test.index, label_predict, color="cornflowerblue",
         label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
