import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

#訓練データ、テストデータ
train_csv = pd.read_csv("data/train.csv")
test_csv = pd.read_csv("data/test.csv")

#データ選択
train_data = train_csv.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
test_data = test_csv.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

#欠損データ処理（訓練、テスト）
train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())
train_data["Embarked"] = train_data["Embarked"].fillna(train_data["Embarked"].mode())
test_data["Age"] = test_data["Age"].fillna(test_data["Age"].median())
test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].mean())


#数値変換　c:1、Q:1、S2
train_012 = pd.DataFrame(train_data["Embarked"])
test_012 = pd.DataFrame(test_data["Embarked"])
#訓練
le = LabelEncoder()
le.fit(train_012)
train_data['Embarked'] = le.transform(train_012)
#テスト
LE = LabelEncoder()
LE.fit(test_012)
test_data['Embarked'] = LE.transform(test_012)

#数値変換 0、1
train = pd.get_dummies(train_data, drop_first=True, prefix='', prefix_sep='')
test = pd.get_dummies(test_data, drop_first=True, prefix='', prefix_sep='')

#変数
x_train = train.drop(columns="Survived")
y_train = train["Survived"]


model = LogisticRegression()
model.fit(x_train, y_train)

