# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
```
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.
```

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1) (1).csv",na_values=[ " ?"])
data
```

<img width="1130" height="492" alt="image" src="https://github.com/user-attachments/assets/c4a97c25-2468-46a2-830f-9191a4d4cc36" />

data.isnull().sum()

<img width="184" height="419" alt="image" src="https://github.com/user-attachments/assets/57b725c6-839a-4a74-8ad3-c8e37db6fb0d" />

```
missing=data[data.isnull().any(axis=1)]
missing
```

<img width="1102" height="350" alt="image" src="https://github.com/user-attachments/assets/fbaaa6d3-aa58-44ad-8247-8f1e5105f9f6" />

```
data2=data.dropna(axis=0)
data2
```

<img width="1124" height="492" alt="image" src="https://github.com/user-attachments/assets/4f219a4e-b055-48a2-96df-4464ba5ae0d9" />

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

<img width="951" height="284" alt="image" src="https://github.com/user-attachments/assets/5e91d13b-ef16-4a7a-8e23-479a7b873d0d" />

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

<img width="261" height="359" alt="image" src="https://github.com/user-attachments/assets/4f796c35-6382-4867-9bda-65a889ac0708" />

data2

<img width="1027" height="336" alt="image" src="https://github.com/user-attachments/assets/e3365112-b22f-4932-906b-f7e7bb571bab" />

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

<img width="1159" height="383" alt="image" src="https://github.com/user-attachments/assets/2c1b96a8-7d60-4ae3-9afd-20fcfa82ff57" />

```
columns_list=list(new_data.columns)
print(columns_list)
```

<img width="1157" height="15" alt="image" src="https://github.com/user-attachments/assets/860e88b8-825f-4567-9427-40a22bdfa240" />

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

<img width="1197" height="88" alt="image" src="https://github.com/user-attachments/assets/31875aca-3750-4703-a7cc-1f6fad36d80a" />

```
y=new_data['SalStat'].values
print(y)
```

<img width="435" height="50" alt="image" src="https://github.com/user-attachments/assets/72a23757-2b09-419e-9502-89cda47b3f8e" />

```
x=new_data[features].values
print(x)
```

<img width="280" height="115" alt="image" src="https://github.com/user-attachments/assets/3bd5da24-dbaf-42fd-80f5-c32c07b7351a" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

<img width="359" height="48" alt="image" src="https://github.com/user-attachments/assets/38ce5850-8661-463f-ada2-f59f7079be8b" />

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```

<img width="363" height="25" alt="image" src="https://github.com/user-attachments/assets/7835228b-c39a-4c57-bb8e-2da68dd42539" />

print("Misclassified Samples : %d" % (test_y !=prediction).sum())

<img width="434" height="39" alt="image" src="https://github.com/user-attachments/assets/8e1a8d73-5702-49b3-b00e-419c2eaafa47" />

data.shape

<img width="195" height="28" alt="image" src="https://github.com/user-attachments/assets/983986e4-0d68-498c-87d2-c69fa413adb1" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

<img width="1160" height="81" alt="image" src="https://github.com/user-attachments/assets/ba184402-47f7-4c4e-85c7-6467acde2eca" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

tips.time.unique()

<img width="477" height="48" alt="image" src="https://github.com/user-attachments/assets/1f4a617b-84c8-43e3-a819-282920c695d1" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

<img width="203" height="72" alt="image" src="https://github.com/user-attachments/assets/c2ffcd9e-a0b4-412f-b2f8-c06cf727f4e3" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

<img width="418" height="40" alt="image" src="https://github.com/user-attachments/assets/52483c13-57dc-412f-85bb-f7534b7eddc1" />



# RESULT:

```
       Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
save the data to a file is been executed
```
