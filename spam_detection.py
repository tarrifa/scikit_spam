import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

##Load Dataset
dataframe = pd.read_csv("spam.csv")
print(dataframe.head())

##Split in to Training and Test Data
x = dataframe["Email Text"]
y = dataframe["Label"]

x_train, y_train = x[0:4457], y[0:4457]
x_test, y_test = x[4457:], y[4457:]

##Extract Features
cv = CountVectorizer()
features = cv.fit_transform(x_train)

##Build a model
tuned_parameters = {'kernel':['linear', 'rbf'], 'gamma':[1e-3, 1e-4], 
                    'C':[1,10,100,1000]}

model = GridSearchCV(svm.SVC(), tuned_parameters)

model.fit(features, y_train)

##Test Accuracy
features_test = cv.transform(x_test)

print("Accuracy of model is: ", model.score(features_test, y_test))



