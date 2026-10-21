import pandas as pd 
from matplotlib import pyplot as plt 
import io
from google.colab import files
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report 

# Upload the CSV file directly from your local machine
print("Please upload 'insurance_data.csv' when prompted:")
uploaded = files.upload()

# Dynamically get the uploaded filename
filename = list(uploaded.keys())[0]

# Read the uploaded CSV file
df = pd.read_csv(io.BytesIO(uploaded[filename])) 
display(df.head())
 
plt.scatter(df.age,df.bought_insurance,marker='+',color='red') 
plt.show() 

x_train,x_test,y_train,y_test=train_test_split(df[['age']],df.bought_insurance, train_size=0.8, random_state=42) 
model=LogisticRegression() 
model.fit(x_train,y_train) 

y_predicted=model.predict(x_test) 
print("Prediction probabilities:\n", model.predict_proba(x_test)) 
print("Model score:", model.score(x_test,y_test)) 

classification_rep=classification_report(y_test,y_predicted) 
print("Classification Report:\n", classification_rep)
