#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[3]:


pip install streamlit scikit-learn pandas numpy


# In[6]:


import joblib
import streamlit as st


# In[214]:


data=pd.read_csv(r"C:\Users\gulnar\Desktop\Project\hr_emp_churn.csv")


# In[215]:


data.head()


# In[216]:


data.shape


# In[217]:


data.isnull().sum()


# In[218]:


data.isnull().sum()/data.shape[0]*100


# # Elave etdiyim codlar
# 

# In[219]:


data = data.drop(columns=['enrollee_id'])


# In[220]:


data.duplicated().sum()


# In[221]:


data.drop_duplicates(inplace=True)
data.duplicated().sum()


# In[222]:


data.describe()


# In[223]:


for column in data.columns:
    print(f"Value counts for column '{column}':")
    print(data[column].value_counts())
    print("\n")


# In[224]:


data['gender'].value_counts()


# In[225]:


data['gender'].fillna("unidentified",inplace=True)


# In[226]:


data['gender'].value_counts()


# In[227]:


data['enrolled_university'].value_counts()


# In[228]:


data['enrolled_university'].fillna(data['enrolled_university'].mode()[0],inplace=True)


# In[229]:


data['enrolled_university'].value_counts()


# In[230]:


data['education_level'].value_counts()


# In[231]:


data['education_level'].fillna(data['education_level'].mode()[0],inplace=True)


# In[232]:


data['education_level'].value_counts()


# In[233]:


data['major_discipline'].value_counts()


# In[234]:


data['major_discipline'].fillna(data['major_discipline'].mode()[0],inplace=True)


# In[235]:


data['major_discipline'].value_counts()


# In[236]:


data['experience'].value_counts()


# In[237]:


data.dropna(subset=['experience'],inplace=True)


# In[238]:


data.loc[data['experience'] == '>20', 'experience'] = '21'
data.loc[data['experience'] == '<1', 'experience'] = '0'
data['experience'] = data['experience'].astype(int)
data['experience'].unique()


# In[239]:


data['company_size'].value_counts()


# In[240]:


data['company_size'].fillna("unidentified",inplace=True)


# In[241]:


data['company_type'].value_counts()


# In[242]:


data['company_type'].fillna("unidentified",inplace=True)


# In[243]:


data['last_new_job'].value_counts()


# In[244]:


data['last_new_job'].fillna(data['last_new_job'].mode()[0],inplace=True)


# In[245]:


data.isnull().sum()


# In[246]:


data.info()


# In[247]:


data.head()


# In[248]:


data.info()


# In[249]:


def find_outliers_iqr(dataframe):
    outliers_count = {}
    
    for column in dataframe.select_dtypes(include=['number']).columns:
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]
        outliers_count[column] = len(outliers)
    
    return outliers_count

outliers_count = find_outliers_iqr(data)
print(outliers_count)


# In[250]:


sns.boxplot(data['city_development_index'])


# In[251]:


sns.boxplot(data['training_hours'])


# In[252]:


data['target'].value_counts()


# In[253]:


data['target'].value_counts(normalize=True)*100


# In[254]:


sns.pairplot(data)


# In[255]:


data.head()


# In[256]:


data['city'].value_counts()


# In[257]:


data['city'].nunique()


# In[258]:


data.drop('city',axis=1,inplace=True)


# In[259]:


data.head()


# In[260]:


sns.countplot(x=data['gender'],hue=data['target'])


# In[261]:


sns.countplot(x=data['relevent_experience'],hue=data['target'])


# In[262]:


sns.countplot(x=data['enrolled_university'],hue=data['target'])


# In[263]:


sns.countplot(x=data['education_level'],hue=data['target'])


# In[264]:


sns.countplot(x=data['major_discipline'],hue=data['target'])


# In[265]:


data['company_size'].unique()


# In[266]:


df_target_0 = data[data['target'] == 0]
df_target_1 = data[data['target'] == 1]


company_size_0 = df_target_0['company_size'].value_counts(normalize=True)
company_size_1 = df_target_1['company_size'].value_counts(normalize=True)


fig, axes = plt.subplots(1, 2, figsize=(12, 6))


axes[0].pie(company_size_0, labels=company_size_0.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"))
axes[0].set_title('Company Size (Target = 0)')


axes[1].pie(company_size_1, labels=company_size_1.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"))
axes[1].set_title('Company Size (Target = 1)')

plt.show()


# In[267]:


data.info()


# In[268]:


data['last_new_job'].value_counts()


# In[269]:


data.loc[data['last_new_job'] == '>4', 'last_new_job'] = '5'
data.loc[data['last_new_job'] == 'never', 'last_new_job'] = '0'
data['last_new_job'] = data['last_new_job'].astype(int)
data['last_new_job'].unique()


# In[270]:


data.head()


# In[271]:


num_data=data.select_dtypes(['float64','int64','int32'])
correlation_matrix = num_data.corr()


plt.figure(figsize=(10, 8))  
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='black')
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[272]:


data.info()


# In[273]:


obj_cols=data.select_dtypes(include='object').columns.to_list()
for i in obj_cols:
    print(data[i].value_counts())
    print()


# In[274]:


df_encoded=pd.get_dummies(data,columns=obj_cols,drop_first=True,dtype='int')


# In[275]:


df_encoded.head()


# In[276]:


df_encoded['target'].value_counts(normalize=True)*100


# ## Scaling

# In[277]:


from sklearn.model_selection import train_test_splitX=df_encoded.drop('target',axis=1)
y=df_encoded[['target']]
X_train,X_test,y_train,y_test=train_test_split(X,y.values.ravel(),test_size=0.2,random_state=42,stratify=y)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## Feature Selection

# In[281]:


len(X_train.columns)


# In[282]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE

best_features = RFE(estimator=ExtraTreesClassifier(n_estimators=100), n_features_to_select=12)
best_features.fit(X_train_scaled, y_train)
selected_features = X.columns[best_features.support_]

print(f"Number of selected features: {len(selected_features)}")
print(f"Selected features : {list(selected_features)}")


# ## LogisticRegression

# In[283]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    make_scorer
)
X_train_selected = best_features.transform(X_train_scaled)
X_test_selected = best_features.transform(X_test_scaled)


clf = LogisticRegression(max_iter=5000, random_state=42)


scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='binary'),
    'recall': make_scorer(recall_score, average='binary'),
    'f1': make_scorer(f1_score, average='binary')
}

cv_results = cross_validate(clf, X_train_selected, y_train, cv=5, scoring=scoring)

print("Cross-Validation Results:")
print(f"Accuracy: {np.mean(cv_results['test_accuracy']):.3f} (+/- {np.std(cv_results['test_accuracy']):.3f})")
print(f"Precision: {np.mean(cv_results['test_precision']):.3f} (+/- {np.std(cv_results['test_precision']):.3f})")
print(f"Recall   : {np.mean(cv_results['test_recall']):.3f} (+/- {np.std(cv_results['test_recall']):.3f})")
print(f"F1 Score : {np.mean(cv_results['test_f1']):.3f} (+/- {np.std(cv_results['test_f1']):.3f})")


clf.fit(X_train_selected, y_train)


y_pred = clf.predict(X_test_selected)
y_pred_proba = clf.predict_proba(X_test_selected)[:, 1] 


test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, average='binary')
test_recall = recall_score(y_test, y_pred, average='binary')
test_f1 = f1_score(y_test, y_pred, average='binary')

print("Test Data Metrics:")
print(f"Accuracy : {test_accuracy:.3f}")
print(f"Precision: {test_precision:.3f}")
print(f"Recall   : {test_recall:.3f}")
print(f"F1 Score : {test_f1:.3f}")


conf_matrix_test = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Test Data)')
plt.show()


fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()


# ## GradientBoosting

# In[284]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
gbc = GradientBoostingClassifier(random_state=42)



scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='binary'),
    'recall': make_scorer(recall_score, average='binary'),
    'f1': make_scorer(f1_score, average='binary')
}


cv_results = cross_validate(gbc, X_train_selected, y_train, cv=5, scoring=scoring)


print("Cross-Validation Results:")
print(f"Accuracy: {np.mean(cv_results['test_accuracy']):.3f} (+/- {np.std(cv_results['test_accuracy']):.3f})")
print(f"Precision: {np.mean(cv_results['test_precision']):.3f} (+/- {np.std(cv_results['test_precision']):.3f})")
print(f"Recall   : {np.mean(cv_results['test_recall']):.3f} (+/- {np.std(cv_results['test_recall']):.3f})")
print(f"F1 Score : {np.mean(cv_results['test_f1']):.3f} (+/- {np.std(cv_results['test_f1']):.3f})")


gbc.fit(X_train_selected, y_train)


y_pred = gbc.predict(X_test_selected)
y_pred_proba = gbc.predict_proba(X_test_selected)[:, 1]  


test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, average='binary')
test_recall = recall_score(y_test, y_pred, average='binary')
test_f1 = f1_score(y_test, y_pred, average='binary')

print("Test Data Metrics:")
print(f"Accuracy : {test_accuracy:.3f}")
print(f"Precision: {test_precision:.3f}")
print(f"Recall   : {test_recall:.3f}")
print(f"F1 Score : {test_f1:.3f}")


conf_matrix_test = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Test Data)')
plt.show()


fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()


# ## RandomForest

# In[285]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(random_state=42)



scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='binary'),
    'recall': make_scorer(recall_score, average='binary'),
    'f1': make_scorer(f1_score, average='binary')
}


cv_results = cross_validate(rf_clf, X_train_selected, y_train, cv=5, scoring=scoring, return_train_score=False)


print("Cross-Validation Results:")
print(f"Accuracy: {np.mean(cv_results['test_accuracy']):.3f} (+/- {np.std(cv_results['test_accuracy']):.3f})")
print(f"Precision: {np.mean(cv_results['test_precision']):.3f} (+/- {np.std(cv_results['test_precision']):.3f})")
print(f"Recall   : {np.mean(cv_results['test_recall']):.3f} (+/- {np.std(cv_results['test_recall']):.3f})")
print(f"F1 Score : {np.mean(cv_results['test_f1']):.3f} (+/- {np.std(cv_results['test_f1']):.3f})")


rf_clf.fit(X_train_selected, y_train)


y_pred = rf_clf.predict(X_test_selected)
y_pred_proba = rf_clf.predict_proba(X_test_selected)[:, 1]  


test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, average='binary')
test_recall = recall_score(y_test, y_pred, average='binary')
test_f1 = f1_score(y_test, y_pred, average='binary')

print("Test Data Metrics:")
print(f"Accuracy : {test_accuracy:.3f}")
print(f"Precision: {test_precision:.3f}")
print(f"Recall   : {test_recall:.3f}")
print(f"F1 Score : {test_f1:.3f}")


conf_matrix_test = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Test Data)')
plt.show()


fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()


# ## KNN

# In[286]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=6)



scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='binary'),
    'recall': make_scorer(recall_score, average='binary'),
    'f1': make_scorer(f1_score, average='binary')
}


cv_results = cross_validate(knn, X_train_selected, y_train, cv=5, scoring=scoring, return_train_score=False)


print("Cross-Validation Results:")
print(f"Accuracy: {np.mean(cv_results['test_accuracy']):.3f} (+/- {np.std(cv_results['test_accuracy']):.3f})")
print(f"Precision: {np.mean(cv_results['test_precision']):.3f} (+/- {np.std(cv_results['test_precision']):.3f})")
print(f"Recall   : {np.mean(cv_results['test_recall']):.3f} (+/- {np.std(cv_results['test_recall']):.3f})")
print(f"F1 Score : {np.mean(cv_results['test_f1']):.3f} (+/- {np.std(cv_results['test_f1']):.3f})")


knn.fit(X_train_selected, y_train)


y_pred = knn.predict(X_test_selected)
y_pred_proba = knn.predict_proba(X_test_selected)[:, 1]  


test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, average='binary')
test_recall = recall_score(y_test, y_pred, average='binary')
test_f1 = f1_score(y_test, y_pred, average='binary')

print("Test Data Metrics:")
print(f"Accuracy : {test_accuracy:.3f}")
print(f"Precision: {test_precision:.3f}")
print(f"Recall   : {test_recall:.3f}")
print(f"F1 Score : {test_f1:.3f}")


conf_matrix_test = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Test Data)')
plt.show()


fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()


# In[ ]:





# In[ ]:




