import pandas as pd
from sklearn.preprocessing import LabelEncoder

#訓練データ、テストデータ
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

#欠損データ処理（訓練、テスト）
train_data["Age"] = train_data["Age"].fillna(train_data["Age"].mean())
train_data["Embarked"] = train_data["Embarked"].fillna(train_data["Embarked"].mode())
test_data["Age"] = test_data["Age"].fillna(test_data["Age"].mean())
test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].mean())

#数値変換　c:1、Q:1、S2
train_em = pd.DataFrame(train_data["Embarked"])
test_em = pd.DataFrame(test_data["Embarked"])
#訓練
le = LabelEncoder()
le.fit(train_em)
train_data['Embarked'] = le.transform(train_em)
print(train_data)
#テスト
LE = LabelEncoder()
LE.fit(test_em)
test_data['Embarked'] = LE.transform(test_em)
print(test_data)

"""
#数値変換 0、1
train = pd.get_dummies(train_data)
test = pd.get_dummies(test_data)
"""