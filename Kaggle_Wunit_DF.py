import pandas as pd

# arranging the database

salesE = pd.read_csv('C:\\Users\Samsung\Desktop\Kaggle\sales_train_evaluation.csv',
                     encoding="utf8")
print(salesE.shape)
salesV = pd.read_csv('C:\\Users\Samsung\Desktop\Kaggle\sales_train_validation.csv',
                     encoding="utf8")
salesE = pd.concat([salesV,salesE],ignore_index=True,sort=False)
print(salesE.shape)

df = pd.read_csv("KaggleSalesRaw.csv")
print(df.head())

columns = ["id"]
for i in range(1,29):
    columns.append("F"+str(i))

print(columns)

df2 = {"id":list(salesE["id"])}

for i in range(28):
    df2[columns[(i+1)]] = list(df.iloc[:,i].astype(int))

df2=pd.DataFrame(data=df2,columns=columns)
df2.to_csv("KaggleSalesInt.csv",index=False)
