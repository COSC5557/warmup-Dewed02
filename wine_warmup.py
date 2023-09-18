import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Read in red wine dataset
wine = pd.read_csv(r'C:\Users\Derek\OneDrive\Desktop\wine+quality\winequality-red.csv', sep=';', engine='python', header=0)

# Assign the 'Quality' column to the y variable
y = wine.quality
# Assign the rest of the colums to the x variable
x = wine.drop(['quality'], axis=1)

# print(wine.head())

# Some Visualization of Features affect on wine quality
# pH = x.pH
# volatile_acidty = x['volatile acidity']
# plt.stem(volatile_acidty, y, label='Volatile Acidity')
# plt.title("Wine Qualtiy and Volatile Acidity")
# plt.legend()
# plt.show()

# Split data into train and test subsets 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Create model
model = linear_model.LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(model.score(x, y))

# Normalize feature data
# Appears to have little to now affect on SGD models ability to generalize the feature from this dataset 
# Also appears to have little affect on Logistic regression model, however it does result in: STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# SGD Classifier
# Currently the accuary score is varying between 0.01 - 0.40
clf = SGDClassifier(loss='squared_error')
clf.fit(x_train, y_train)

y_pred_clf = clf.predict(x_test)
print("SGD Accuary: {:.2f}".format(accuracy_score(y_test, y_pred_clf)))

# Logistic Regression Classifier
# Provides a better accuray score than SGD classifier, ~0.6, however unsure if this is meaniful. 
lgClf = LogisticRegression(random_state=0, max_iter=1599).fit(x_train, y_train)
y_pred_lgClf = clf.predict
print("Logistic Regression Accuary: {:.2f}".format(lgClf.score(x_train, y_train)))


# I have consulted a variety of sources while working on this warmup including: 
# https://www.youtube.com/watch?v=EuBBz3bI-aA&t=159s, https://www.youtube.com/watch?v=7ArmBVF2dCs
# https://realpython.com/linear-regression-in-python/, https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
# At this point I have created a plot for each of the features in the dataset to see if I could see which of the features had the greatest impact on the quality of the wine
# however, one of the videos that I linked stated that linear regression models only take into account the features with the greatest impact and for all intents and purposes
# ignores all other features. I'm not sure if that is the case for linear classifiers, but I could not see any features that had a significat impact on wine quality. 
# I am unsure how to preprocess the feature data in this dataset, at the moment I have all feature data normalized so that it appears normally distributed. However, 
# both the SGD and Logistic Regression classfier produce similar results with or without the preprocessing, so perhaps the data is already normally distributed? 
# I intially tried SGD because I had used models that use this optimization algorithm for the computer vision project I am working on at ARCC, however, it produced poor
# and quite variable results so I tried a logistic regression model instead to see if it would produce better results. I would like to continue working on this warm-up 
# but I am unsure of what more I can do to produce better prediction results. 