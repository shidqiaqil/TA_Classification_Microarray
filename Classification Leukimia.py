#!/usr/bin/env python
# coding: utf-8

# In[124]:


#load dataset
import pandas as pd
import numpy as np


# In[131]:


data = pd.read_csv(r'D:\GitHub\TA_Classification_Microarray\dataset\Leukimia\MLL_train.data', header=-1)

data=data.drop(data.index[[20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]])


# In[132]:


x = data.iloc[:, :-1] #mengambil semua row dan kolom kecuali kolom terakhir
y = data.iloc[:, -1:]#mengambil semua baris namun hanya kolom terakhir
# scaler = StandardScaler()
# X_std = scaler.fit_transform(X_std,x)

print(data)


# In[133]:


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# In[134]:


x= NormalizeData(x)
y=np.ravel(y)
print(y)


# In[135]:


from sys import stdout
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


# In[136]:


# Split data to train and test on 50-50 ratio
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=None)


# In[137]:


pls = PLSRegression(n_components=27)
pls.fit(X_train, X_test)
X_pls = pls.fit_transform(X_train, X_test)
x2 =pls.transform(x)


# In[138]:


x2=pd.DataFrame(x2)
print(x2)
#x2= NormalizeData(x2)
#print(X_pls)
#two_arrays = X_pls
#datapls = np.hstack(two_arrays)
#np.savetxt('lungcancerpls111.csv', datapls, delimiter=',')


# In[139]:


from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
j=2
k=2
#i=2


# In[146]:


#create a new KNN model
while k<10:
    i=2;
    knn = KNeighborsClassifier(n_neighbors= k)
    while i<10:
        cv_scores = cross_val_score(knn, x2,y , cv= i)
        a = cv_scores
        #print(cv_scores)
        print("cv_scores mean k=",k,"fold ",i,": {}".format(np.mean(cv_scores)*100),"%")
        i+=1
    k+=1
#knn.fit(y_c,)
#y_pred= knn.predict(X_test)


# In[145]:


from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
j=2
k=1

while j<10:
    classifier = SVC(kernel='linear', C=10, gamma=0.1)
    classifier.fit(X_train,y_train)
    cv_scores = cross_val_score(classifier , x2,y , cv=j)
    #print(cv_scores)
    print("cv_scores mean:{}".format(np.mean(cv_scores)*100),"%")
    j+=1


# In[142]:


data2 = pd.read_csv(r'D:\GitHub\TA_Classification_Microarray\dataset\Leukimia\MLL_test.data', header=-1)
print(data2)


# In[ ]:





# In[ ]:




