import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

#訓練データ、テストデータ
train_csv = pd.read_csv("data/train.csv")
test_csv = pd.read_csv("data/test.csv")

class Titanic:
    def __init__(self, train_csv, test_csv):

        #データ選択
        self.train_data = train_csv.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
        self.test_data = test_csv.drop(columns=["Name", "Ticket", "Cabin"])

        #訓練、テストデータ完成
        self.train = self.process_data(self.train_data)
        self.test = self.process_data(self.test_data)

        #ロジスティック回帰
        self.logistic = LogisticRegression()


    #データ処理
    def process_data(self, df):

        #欠損データ処理
        df["Age"] = df["Age"].fillna(df["Age"].median())
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode())
        df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

        #カテゴリー変数をダミー変数へ
        df_proc = self.dummies(df)

        return df_proc


    #ダミー変数
    def dummies(self, df):

        #数値変換　c:1、Q:1、S:2
        df_le = pd.DataFrame(df["Embarked"])
        le = LabelEncoder()
        df['Embarked'] = le.fit_transform(df_le)

        #数値変換 female:0、male:1
        df_dum = pd.get_dummies(df, drop_first=True, prefix='', prefix_sep='')

        return df_dum
    

    #訓練用データセット
    def dataset_train(self, dataname):
        if dataname == "train":
            #変数
            x_train = self.train.drop(columns="Survived")
            y_train = self.train["Survived"]
            return x_train, y_train

    #テスト用データセット
    def dataset_test(self, dataname):
        if dataname == "test":
            x_test = self.test.drop(columns="PassengerId")
            return x_test
    

    #モデルの訓練
    def model(self, x, y):
        self.logistic.fit(x, y)


    #生存者予測
    def survived_predict(self, x):
        y = self.logistic.predict(x)
        return y


    #結果確認
    def result(self, y):
        self.test['Survived'] = y
        print(self.test)

        #ファイル出力
        solution = self.test.loc[:, ["PassengerId", "Survived"]]
        solution.to_csv("titanic.csv", index=False)


#メイン
def main(titanic):
    #学習用x,y
    (x, y) = titanic.dataset_train("train")
    #学習
    titanic.model(x, y)

    #テスト用x
    x_test = titanic.dataset_test("test")
    #予測
    survived = titanic.survived_predict(x_test)
    #結果
    titanic.result(survived)
    

if __name__ == "__main__":
    titanic = Titanic(train_csv, test_csv)
    main(titanic)