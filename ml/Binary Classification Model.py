import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score 
 
iris=load_iris() 
X=iris.data 
y=iris.target 
 
y=np.where(y==0,0,1) 
 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42) 
 
kernels=['linear','poly','rbf','sigmoid'] 
metrics={"Accuracy":[],"Precision":[],"Recall":[],"F1 Score":[]} 
 
for k in kernels: 
    svm=SVC(kernel=k,random_state=42) 
    svm.fit(X_train,y_train) 
    y_pred=svm.predict(X_test) 
 
    acc=accuracy_score(y_test,y_pred) 
    prec=precision_score(y_test,y_pred) 
    rec=recall_score(y_test,y_pred) 
    f1 =f1_score(y_test,y_pred) 
 
    metrics["Accuracy"].append(acc) 
    metrics["Precision"].append(prec) 
    metrics["Recall"].append(rec) 
    metrics["F1 Score"].append(f1) 
 
    print(f"\nkernel:{k}") 
    print(f"Accuracy:{acc:.4f}") 
    print(f"Precision:{prec:.4f}") 
    print(f"Recall:{rec:.4f}") 
    print(f"F1 Score:{f1:.4f}") 
 
x=np.arange(len(kernels)) 
width=0.2 
 
plt.bar(x - 0.3, metrics["Accuracy"],width,label='Accuracy') 
plt.bar(x - 0.1,metrics["Precision"],width,label='Precision') 
plt.bar(x + 0.1,metrics["Recall"],width,label='Recall') 
plt.bar(x + 0.3,metrics["F1 Score"],width,label='F1 Score') 
 
plt.xticks(x,kernels) 
plt.ylabel("Score") 
plt.title("SVM Performance with Different Kernels") 
plt.legend() 
plt.show()
