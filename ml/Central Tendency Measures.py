import pandas as pd 
import random 
 
dataset={ 
"marrital_status": random.choices(["Yes","No"],k=600), 
"is_graduate": random.choices(["Yes","No"],k=600), 
"Income": [random.randint(20000,100000) for _ in range(600)], 
"loan_amount":[random.randint(1000,50000) for _ in range(600)], 
"term_months": random.choices([12,24,36,48,60], k=600), 
"Age": [random.randint(20,60) for _ in range(600)], 
"sex": random.choices(["F","M"], k=600), 
"approval_status": random.choices(["Yes", "No"], k=600) 
} 
 
data=pd.DataFrame(dataset) 
data.to_csv(r"C:\Users\majeed\Downloads\dataset\dataset.csv", index=False) 
 
print("mean:", round(data['Income'].mean(),3)) 
print("median:", data['Income'].median()) 
print("SD", data['Income'].std()) 
print("var", data['Income'].var()) 
print("mode", data['Income'].mode()[0])

