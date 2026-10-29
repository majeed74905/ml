import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report 
df=pd.read_csv(r"C:\Users\majeed\Downloads\ml-main\ml-main\dataset\insurance_data.csv") 
df.head() 
 
plt.scatter(df.age,df.bought_insurance,marker='+',color='red') 
plt.show() 
x_train,x_test,y_train,y_test=train_test_split(df[['age']],df.bought_insurance, 
train_size =0.8) 
model=LogisticRegression() 
model.fit(x_train,y_train) 
y_predicted=model.predict(x_test) 
model.predict_proba(x_test) 
model.score(x_test,y_test) 
classification_rep=classification_report(y_test,y_predicted) 
print(classification_rep)
