import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# CSV読み込み
df = pd.read_csv('data/wine.csv')

# 説明変数
X = df[['alcohol']]

# 目的変数
y = df['quality']

# モデル作成
model = LinearRegression()
model.fit(X, y)

# 予測
y_pred = model.predict(X)

# グラフ作成
plt.scatter(X, y)
plt.plot(X, y_pred)

plt.xlabel('alcohol')
plt.ylabel('quality')
plt.title('Linear Regression')

# wine-analysis直下に保存
plt.savefig('linear_regression.png')

plt.show()