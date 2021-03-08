# Load the relevent Python modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

# load and investigate the data
df = pd.read_csv('tennis_stats.csv')

print(df.head())
print(df.columns)
print(df.describe())

print("There are {} unique players is this data base.".format(df.Player.nunique()))
print('The years covered in this data base are: {}'.format(sorted(df.Year.unique())))


# exploratory analysis:
print(df.corr())
plt.scatter(df.Wins, df.BreakPointsFaced)
plt.xlabel('Wins')
plt.ylabel('Break Points Faced')
plt.title('Wins vs Break Points Faced')
plt.show()

plt.scatter(df.Wins, df.BreakPointsOpportunities)
plt.xlabel('Wins')
plt.ylabel('Break Points Opportunities')
plt.title('Wins vs Break Points Opportunities')
plt.show()

plt.scatter(df.Wins, df.FirstServe)
plt.xlabel('Wins')
plt.ylabel('First Serve')
plt.title('Wins vs First Serve')
plt.show()

plt.scatter(df.Wins, df.TotalServicePointsWon)
plt.xlabel('Wins')
plt.ylabel('Total Service Points Won')
plt.title('Wins vs Total Service Points Won')
plt.show()


# train and test a linear regression model using the break points opportunities and wins columns from the data frame
features = df.BreakPointsOpportunities
wins = df.Wins

x_train, x_test, y_train, y_test = train_test_split(
    features, wins, test_size=0.2, random_state=1)

x_test = x_test.values.reshape(-1, 1)
x_train = x_train.values.reshape(-1, 1)
model = LinearRegression()

model.fit(x_train, y_train)

prediction_score = model.score(x_test, y_test)
print(prediction_score)

prediction = model.predict(x_test)

plt.scatter(y_test, prediction, alpha=0.4)
plt.xlabel('Actual Wins')
plt.ylabel('Predicted Wins')
plt.title('Actual Wins vs Predicted Wins - 1 Feature')
plt.show()

# train and test a linear regression model using the Break Points Faced and Wins columns from the data frame

features = df.BreakPointsFaced
wins = df.Wins

x_train, x_test, y_train, y_test = train_test_split(
    features, wins, test_size=0.2, random_state=1)

x_test = x_test.values.reshape(-1, 1)
x_train = x_train.values.reshape(-1, 1)

model = LinearRegression()

model.fit(x_train, y_train)

prediction_score = model.score(x_test, y_test)
print(prediction_score)

prediction = model.predict(x_test)

plt.scatter(y_test, prediction, alpha=0.4)
plt.xlabel('Actual Wins')
plt.ylabel('Predicted Wins')
plt.title('Actual Wins vs Predicted Wins')
plt.show()

# train and test a linear regression model which uses multiple features to predict wins
features = df[['BreakPointsOpportunities', 'FirstServePointsWon']]
wins = df.Wins

x_train, x_test, y_train, y_test = train_test_split(
    features, wins, test_size=0.2, random_state=1)


model = LinearRegression()

model.fit(x_train, y_train)

prediction_score = model.score(x_test, y_test)
print(prediction_score)

prediction = model.predict(x_test)

plt.scatter(y_test, prediction, alpha=0.4)
plt.xlabel('Actual Wins')
plt.ylabel('Predicted Wins')
plt.title('Actual Wins vs Predicted Wins')
plt.show()
