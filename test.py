import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

#文字列をintに
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

#データ確認
print(train.head())
print(test.head())

#統計情報
print(train.describe())
print(test.describe())

#欠損確認
print(train.isnull().sum())
print(test.isnull().sum())

#データ分布
train.hist(figsize=(12, 10))
plt.savefig("train.png")

#データ処理
train = train.fillna({"Age" : 30, "Embarked" : "0"})
test= test.fillna({"Age" : 30, "Fare" : test.Fare.mean()})

#モデル構築
x = train.drop(columns=["PassengerId", "Survived", "Name", "Ticket", "Cabin"])
y = train["Survived"]

model = LinearRegression()
model.fit(x, y)