import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score

# 데이터 파일 읽기
airbnb=pd.read_json('/AB_NYC_2019.json')

# 데이터 컬럼 확인
airbnb.columns

# neighbourhood_group 데이터 시각화
sns.countplot(airbnb['neighbourhood_group'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Neighbourhood Group')

# neighbourhood 데이터 시각화
sns.countplot(airbnb['neighbourhood'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(25,6)
plt.title('Neighbourhood')

# room_type 데이터 시각화
sns.countplot(airbnb['room_type'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Restaurants delivering online or Not')

# price 데이터 시각화
sns.countplot(airbnb['price'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(25,6)
plt.title('Price')

# number_of_reviews 데이터 시각화
sns.countplot(airbnb['number_of_reviews'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(25,6)
plt.title('number_of_reviews')

# 데이터 전처리
airbnb.drop(['id', 'name', 'host_id', 'host_name', 'latitude', 'longitude',
       'minimum_nights', 'last_review',
       'reviews_per_month', 'calculated_host_listings_count',
       'availability_365'], axis=1, inplace=True)
#examing the changes
airbnb.head(5)

# 데이터 인코딩
def Encode(airbnb):
    for column in airbnb.columns[airbnb.columns.isin(['neighbourhood_group', 'neighbourhood', 'room_type', 'price'])]:
        airbnb[column] = airbnb[column].factorize()[0]
    return airbnb

airbnb_en = Encode(airbnb.copy())
airbnb_en.head(15)

# 데이터 간의 상관관계 그래프 시각화
corr = airbnb_en.corr(method='kendall')
plt.figure(figsize=(18,12))
sns.heatmap(corr, annot=True)
airbnb_en.columns

# 데이터 학습
x = airbnb_en.iloc[:,[0,1,3,4]]
y = airbnb_en['number_of_reviews']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=353)
x_train.head()
y_train.head()
x_train.shape

#Linear Regression Model
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=105)
DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train,y_train)
y_predict=DTree.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)
