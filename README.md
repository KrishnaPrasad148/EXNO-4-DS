# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

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
Developed By : Krishna Prasad.S
Register No : 212223230108
```

```
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv('bmi.csv')
df.head()
```
![Screenshot 2024-10-03 081922](https://github.com/user-attachments/assets/ef1ddf4d-4c60-4e51-802f-d7f81afb9da6)

```
df.dropna()
```
![Screenshot 2024-10-03 082121](https://github.com/user-attachments/assets/35071cc8-8952-4058-943d-194034febb72)

```
max_vals = np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![322402860-69a21166-1be8-4779-8e21-b653d7c4f52e](https://github.com/user-attachments/assets/278642b7-1289-4396-b077-c775fc65231a)

```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df[['Height','Weight']] = sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/75cbdaf5-d937-42a2-b8a3-6bf64e3d1f65)

```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Height','Weight']] = scaler.fit_transform(df[['Height','Weight']])
df.head(10)

```
![Screenshot 2024-10-03 084112](https://github.com/user-attachments/assets/e3584486-2eed-4ac9-ab2b-ca75eee5a45d)

```
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
df[['Height','Weight']] = scaler.fit_transform(df[['Height','Weight']])
df
```
![Screenshot 2024-10-03 085506](https://github.com/user-attachments/assets/93bcf711-637c-47e3-a899-d4b705dcaa46)

```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df[['Height','Weight']] = scaler.fit_transform(df[['Height','Weight']])
df
```
![Screenshot 2024-10-03 085632](https://github.com/user-attachments/assets/1a18c4f6-6a21-4726-8bb2-6e3833428eb9)

```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df[['Height','Weight']] = scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![Screenshot 2024-10-03 085739](https://github.com/user-attachments/assets/da09a37f-58d2-460a-84bf-4b1463cb20d3)

```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![Screenshot 2024-10-03 100306](https://github.com/user-attachments/assets/cc8e605f-3663-4216-aed1-4c50f2cfeb30)

```
data.isnull().sum()
```
![Screenshot 2024-10-03 085959](https://github.com/user-attachments/assets/6995be1d-5ec5-42f3-adba-de87bed42bb1)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![Screenshot 2024-10-03 100509](https://github.com/user-attachments/assets/d5d11c0d-43e0-4e7b-8550-0be0d8c7ab83)

```
data2=data.dropna(axis=0)
data2
```
![Screenshot 2024-10-03 100522](https://github.com/user-attachments/assets/acd7cb7e-b6cb-49b6-9872-c04208173d68)

```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![Screenshot 2024-10-03 090204](https://github.com/user-attachments/assets/f0817d43-6808-492c-b31b-a52fb8469a65)

```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![Screenshot 2024-10-03 090232](https://github.com/user-attachments/assets/41c84ccb-64ab-4a49-99ec-0edeb9ca7d56)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![Screenshot 2024-10-03 100624](https://github.com/user-attachments/assets/d323f497-d8d8-40dd-92a7-de5808d061e7)

![Screenshot 2024-10-03 100639](https://github.com/user-attachments/assets/3a2dbc84-05d5-4b64-a317-e3cba4b3efc1)


```
columns_list=list(new_data.columns)
print(columns_list)
```
![Screenshot 2024-10-03 090700](https://github.com/user-attachments/assets/38c4454b-8aac-498e-945e-e11a9014f2a2)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![Screenshot 2024-10-03 090740](https://github.com/user-attachments/assets/cae9fdb4-680d-4db9-8ccc-7bae23edee12)

```
y=new_data['SalStat'].values
print(y)
```
![Screenshot 2024-10-03 090813](https://github.com/user-attachments/assets/fd02e1c7-1a36-4f7d-9688-543245231f09)

```
x=new_data[features].values
print(x)
```
![Screenshot 2024-10-03 090843](https://github.com/user-attachments/assets/0858f598-e7ae-4421-b334-b45c608ccaee)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![Screenshot 2024-10-03 090916](https://github.com/user-attachments/assets/1b2a1f80-cc4c-4f03-a571-9e57fba59782)

```
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![Screenshot 2024-10-03 090948](https://github.com/user-attachments/assets/c5fc969f-cf2f-4b3f-8840-f9ec6b7c4819)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![Screenshot 2024-10-03 091019](https://github.com/user-attachments/assets/aacddcaa-003f-4600-b31a-0e25423e21b1)

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![Screenshot 2024-10-03 095523](https://github.com/user-attachments/assets/c710e3a9-b560-4551-80de-9bce35f005ba)

```
data.shape
```
![Screenshot 2024-10-03 095612](https://github.com/user-attachments/assets/9a69279d-6cd9-49da-a1ff-13553c004608)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
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
![Screenshot 2024-10-03 095727](https://github.com/user-attachments/assets/66e9c726-7eaf-4a8d-ad52-687afedc6fbf)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![Screenshot 2024-10-03 095809](https://github.com/user-attachments/assets/9e45ac1a-9cac-40f2-bbe4-0c03d158ab75)

```
tips.time.unique()
```
![Screenshot 2024-10-03 095844](https://github.com/user-attachments/assets/5fc7596a-bbe2-4c7f-8100-404ab421ca64)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![Screenshot 2024-10-03 095930](https://github.com/user-attachments/assets/fe75ab7e-5cfb-4d7a-964c-7b4cfd2e7e17)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/user-attachments/assets/14cee456-2960-4594-9ba5-30fdabc61eac)


# RESULT:
       Thus, Feature selection and Feature scaling has been used on thegiven dataset.
