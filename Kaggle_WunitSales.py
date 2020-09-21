# developed for estimating the unit sales of Walmart retail goods using machine learning

import pandas as pd
import scipy.stats as sp
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.layers import Dense, Embedding, Dropout, Reshape
from tensorflow.keras.models import Sequential
from keras.callbacks import TensorBoard
from time import time
from sklearn.model_selection import train_test_split

# reading the databases
salesE = pd.read_csv('C:\\Users\Samsung\Desktop\Kaggle\sales_train_evaluation.csv',
                     encoding="utf8")
print(salesE.shape)
salesV = pd.read_csv('C:\\Users\Samsung\Desktop\Kaggle\sales_train_validation.csv',
                     encoding="utf8")
salesE = pd.concat([salesV,salesE],ignore_index=True,sort=False)
print(salesE.head())
print(salesE.tail())
calendar = pd.read_csv('C:\\Users\Samsung\Desktop\Kaggle\calendar.csv',encoding="utf8")
print(calendar.shape)

# filtering the date
calendar["date_dt"] = pd.to_datetime(calendar["date"])
cindex = calendar.loc[(calendar["date_dt"].dt.month==6) & (calendar["date_dt"].dt.day>19)].index.values
cindex7 = calendar.loc[(calendar["date_dt"].dt.month==7) & (calendar["date_dt"].dt.day<18)].index.values

cindex = list(cindex)
cindex7 = list(cindex7)
cindex.extend(cindex7)
cindex.sort()
print(len(cindex))

cindex = [x+6 for x in cindex]

sub = {}

for i in range(28):
    cindexNew = cindex[i::28]
    sub[str(i)] = []
    print(cindexNew)

    for j in salesE.index:

        sales5 = list(salesE.iloc[j,cindexNew])
        x = sales5

        std = np.std(x)
        med = np.median(x)

        a = np.array(x)
        upper_quartile = np.percentile(a, 75)
        lower_quartile = np.percentile(a, 25)
        IQR = (upper_quartile - lower_quartile)
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
        xout = []
        for y in a.tolist():
            if y >= quartileSet[0] and y <= quartileSet[1]:
                xout.append(y)

        x=xout

        count = C(x).most_common()

        # linear regression
        if len(x)>4:
            if x[-1] > x[-2] and x[-2] > x[-3] and x[-3] > x[-4]:

                slope, intercept, r_value, p_value, std_err = sp.linregress(range(1, 6), x)
                pred = slope * 6 + intercept
                pred = np.round(pred)

            elif x[-1] < x[-2] and x[-2] < x[-3] and x[-3] < x[-4]:

                slope, intercept, r_value, p_value, std_err = sp.linregress(range(1, 6), x)
                pred = slope * 6 + intercept
                pred = np.round(pred)
                if pred < 0:
                    pred = 0

            elif count[0][1] > 3:
                # pred = max(x,key=x.count)
                pred = count[0][0]
            else:
                pred = np.round((x[-1] + x[-2]) / 2)

        elif count[0][1] > 2:
            pred = count[0][0]
        else:
            if len(x) < 2:
                pred = x[0]
            else:
                pred = np.round((x[-1] + x[-2]) / 2)

        sub[str(i)].append(pred)


df = pd.DataFrame(sub)
print(df.head(n=20))
print(df.shape)

df.to_csv("KaggleSalesRaw.csv",index=False)

X=np.array(X)
Y=np.array(Y)
print(X.shape)

# splitting the data into the train and test
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.20)

dim = x_train.shape[1]

# neural network model
model = Sequential()
model.add(Dense(2, input_dim=dim, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='D:\Tensorboard'.format(time()))

model.fit(x_train, y_train,  # 21,64,57
                  epochs=8, batch_size=32,
                  validation_data=(x_val, y_val),
                  callbacks=[tensorboard])

trues = 0
falses = 0

# testing the model
for i in range(len(x_test)):

    pred = model.predict(np.array([x_test[i]]))
    print(pred)
    pred = np.round(pred)
    pred = int(pred[0][0])
    print(pred)
    print(y_test[i])
    if pred == y_test[i]:
        trues +=1
    else:
        falses += 1

print('Test accuracy:', ((trues/(trues+falses))*100))
