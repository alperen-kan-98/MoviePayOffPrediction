import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Loading the raw dataset
rawMovieData = pd.read_excel('2014 and 2015 CSM dataset.xlsx')

# Start cleaning
# ---> Start with missing values
rawMovieData.isnull().sum()
# ---> In 'Budget' only 1 row is NA so we can drop it
rawMovieData.dropna(subset=['Budget'], inplace =True)
# ---> Budget attribute show in scientific notation so convert to int
rawMovieData['Budget'] = rawMovieData['Budget'].astype(int)
# ---> Next we have 10 missing value in screens
# ---> TODO: Predict the missing values instead of dropping them
rawMovieData.dropna(subset=['Screens'], inplace =True)
# ---> And lastly we have 33 missing in aggregate Followers
rawMovieData.dropna(subset=['Aggregate Followers'], inplace =True)
# ---> Aggregate Followers attribute show in scientific notation so convert to int
rawMovieData['Aggregate Followers'] = rawMovieData['Aggregate Followers'].astype(int)

# ---> Pay off with only ticket = If the movie gain more than its budget with only
# ---> the money which gained with the movie tickets. (Total movie gain is not just ticket gain)
rawMovieData['IsPayOffWithOnlyTicket'] = rawMovieData['Gross'] - rawMovieData['Budget']
# If no its assigned 0
rawMovieData['IsPayOffWithOnlyTicket'][rawMovieData['IsPayOffWithOnlyTicket'] < 0] = 0
# If yes its assigned 1
rawMovieData['IsPayOffWithOnlyTicket'][rawMovieData['IsPayOffWithOnlyTicket'] > 0] = 1

processedData = pd.DataFrame()
processedData['Ratings'] = rawMovieData['Ratings']
processedData['Budget'] = rawMovieData['Budget']
processedData['Screens'] = rawMovieData['Screens']
processedData['Sentiment'] = rawMovieData['Sentiment']
processedData['Views'] = rawMovieData['Views']
processedData['Like-Dislike'] = rawMovieData['Likes'] - rawMovieData['Dislikes']
processedData['Comments'] = rawMovieData['Comments']
processedData['TotalFollowers'] = rawMovieData['Aggregate Followers']

targetData = pd.DataFrame()
targetData['IsPayOffWithOnlyTicket'] = rawMovieData['IsPayOffWithOnlyTicket']
# End of preperation

# Start Scaling
from sklearn.preprocessing import MinMaxScaler
scaledData = pd.DataFrame()
mmScaler = MinMaxScaler()

scaledData['Ratings'] = processedData['Ratings']

temp = processedData['Budget'].values
temp = temp.reshape(-1,1)
scaledData['Budget'] = mmScaler.fit_transform(temp)

temp = processedData['Screens'].values
temp = temp.reshape(-1,1)
scaledData['Screens'] = mmScaler.fit_transform(temp)


scaledData['Sentiment'] = processedData['Sentiment']

temp = processedData['Views'].values
temp = temp.reshape(-1,1)
scaledData['Views'] = mmScaler.fit_transform(temp)

temp = processedData['Like-Dislike'].values
temp = temp.reshape(-1,1)
scaledData['Like-Dislike'] = mmScaler.fit_transform(temp)

temp = processedData['Comments'].values
temp = temp.reshape(-1,1)
scaledData['Comments'] = mmScaler.fit_transform(temp)

temp = processedData['TotalFollowers'].values
temp = temp.reshape(-1,1)
scaledData['TotalFollowers'] = mmScaler.fit_transform(temp)

temp2 = processedData['Ratings'].values
temp2 = temp2*10
temp2 = temp2.astype(int)
temp2 = temp2.reshape(-1,1)
scaledData['Ratings'] = mmScaler.fit_transform(temp2)

temp3 = processedData['Sentiment'].values
temp3 = temp3.reshape(-1,1)
scaledData['Sentiment'] = mmScaler.fit_transform(temp3)

# End of scaling

# ---------->  VISUALIZATIONS  <----------
visualData = scaledData.copy()
visualData['IsPayOffWithOnlyTicket'] = targetData['IsPayOffWithOnlyTicket'].copy()

import seaborn as sns
sns.pairplot(visualData,kind='reg')

sns.heatmap(visualData.corr(), annot = True)

rawMovieData.describe()


# ---------->  VISUALIZATIONS  <----------

calculationsdf = pd.DataFrame()
def calculations(arr):
    TrueNeg = arr[0,0]
    FalseNeg = arr[0,1]
    FalsePos = arr[1,0]
    TruePos = arr[1,1]
    pos = TruePos + FalsePos
    neg = TrueNeg + FalseNeg
    sensitivity = TruePos / (TruePos + FalseNeg)
    specificity = TrueNeg / (TrueNeg + FalsePos)
    precision = TruePos / (TruePos + FalsePos)
    accuracy = sensitivity * ((pos)/(pos+neg)) + specificity * ((neg)/(pos+neg))
    f_measure = (2 * precision * sensitivity) / (precision + sensitivity)
    calcDisc = {
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision": precision,
            "Accuracy": accuracy,
            "F-Measure": f_measure}
    return calcDisc

# Start Data splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(scaledData, targetData, test_size = 0.40)
# End Data splitting

# ---> LOGISTIC REGRESSION <---
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression(random_state = 0)
logReg.fit(x_train,y_train)
y_pred = logReg.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
plt.xlabel("Real Value")
plt.ylabel("Predicted value")
plt.title("LOGISTIC REGRESSION")
plt.show()
logisticCalc= calculations(cm)
# ---> LOGISTIC REGRESSION <---

# ---> DECISION TREE CLASSIFIER <---
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth = 3, criterion = 'entropy', random_state = 0)
dtc.fit(x_train, y_train)
dtPred = dtc.predict(x_test)
cm2 = confusion_matrix(y_test, dtPred)
sns.heatmap(cm2/np.sum(cm2), annot=True, fmt='.2%', cmap='Greens')
plt.xlabel("Real Value")
plt.ylabel("Predicted value")
plt.title("DECISION TREE CLASSIFIER")
plt.show()
dtcCalc= calculations(cm2)
# ---> DECISION TREE CLASSIFIER <---

# ---> KNN CLASSIFIER <---
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) 
knn.fit(x_train, y_train)
knnPred = knn.predict(x_test)
cm3 = confusion_matrix(y_test, knnPred)
sns.heatmap(cm3/np.sum(cm3), annot=True, fmt='.2%', cmap='copper_r')
plt.xlabel("Real Value")
plt.ylabel("Predicted value")
plt.title("KNN CLASSIFIER")
plt.show()
knnCalc= calculations(cm3)
# ---> KNN CLASSIFIER <---

# ---> RANDOM FOREST CLASSIFIER <---
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 50, max_depth = 5, criterion = 'entropy', random_state = 0) # 
rfc.fit(x_train, y_train)
rfcPred = rfc.predict(x_test)
cm4 = confusion_matrix(y_test, rfcPred)
sns.heatmap(cm4/np.sum(cm4), annot=True, fmt='.2%', cmap='binary')
plt.xlabel("Real Value")
plt.ylabel("Predicted value")
plt.title("RANDOM FOREST CLASSIFIER")
plt.show()
randomCalc = calculations(cm4)
# ---> RANDOM FOREST CLASSIFIER <---

# ---> SUPPORT VECTOR CLASSIFIER (LINEAR) <---
from sklearn.svm import SVC
svm = SVC(kernel = 'linear', random_state = 0) 
svm.fit(x_train, y_train)
svcPred = svm.predict(x_test)
cm5 = confusion_matrix(y_test, svcPred)
sns.heatmap(cm5/np.sum(cm5), annot=True, fmt='.2%', cmap='PuBu')
plt.xlabel("Real Value")
plt.ylabel("Predicted value")
plt.title("SUPPORT VECTOR CLASSIFIER (LINEAR)")
plt.show()
svmLCalc = calculations(cm5)
# ---> SUPPORT VECTOR CLASSIFIER (LINEAR) <---

# ---> SUPPORT VECTOR CLASSIFIER (RBF) <---
svm2 = SVC(kernel = 'rbf', random_state = 0) 
svm2.fit(x_train, y_train)
svcPred2 = svm2.predict(x_test)
cm6 = confusion_matrix(y_test, svcPred2)
svmRCalc = calculations(cm6)
sns.heatmap(cm6/np.sum(cm6), annot=True, fmt='.2%', cmap='Reds')
plt.xlabel("Real Value")
plt.ylabel("Predicted value")
plt.title("SUPPORT VECTOR CLASSIFIER (RBF)")
plt.show()
# ---> SUPPORT VECTOR CLASSIFIER (RBF) <---
