#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load dataset
import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv(r'D:\GitHub\TA_Classification_Microarray\dataset\Breast Cancer\breastCancer_train.data', header=-1)


# In[3]:


x = data.iloc[:, :-1] #mengambil semua row dan kolom kecuali kolom terakhir
y = data.iloc[:, -1:]#mengambil semua baris namun hanya kolom terakhir
# scaler = StandardScaler()
# X_std = scaler.fit_transform(X_std,x)
print(data)


# In[4]:


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# In[8]:


x= NormalizeData(x)
y=np.ravel(y)
y=pd.DataFrame(y)
#pd.set_option('display.max_rows', x.shape[0]+1)
print(y)


# In[60]:


x.fillna(x.mean())


# In[7]:


for k in range(x.n_components):
            if np.all(np.dot(Yk.T, Yk) < np.finfo(np.double).eps):
                # Yk constant
                warnings.warn('Y constant at iteration %s' % k)
                break


# In[50]:


from sys import stdout
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


# In[51]:


# Split data to train and test on 50-50 ratio
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=None)


# In[55]:



pls = PLSRegression(n_components=5)
pls.fit(X_train, X_test)
X_pls = pls.fit_transform(X_train, X_test)
x2 =pls.transform(x)


# In[245]:


x2=pd.DataFrame(x2)
print(x2)
#print(X_pls)
#two_arrays = X_pls
#datapls = np.hstack(two_arrays)
#np.savetxt('lungcancerpls111.csv', datapls, delimiter=',')


# In[253]:


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
k=1
#i=2


# In[259]:


#create a new KNN model
while k<10:
    i=2;
    knn = KNeighborsClassifier(n_neighbors= k)
    while i<10:
        cv_scores = cross_val_score(knn, x2,y , cv= i)
        a = cv_scores
        print(cv_scores)
        print("cv_scores mean k=",k,": {}".format(np.mean(cv_scores)))
        i+=1
    k+=1
#knn.fit(y_c,)
#y_pred= knn.predict(X_test)


# In[258]:


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
    print(cv_scores)
    print("cv_scores mean:{}".format(np.mean(cv_scores)))
    j+=1


# In[ ]:





# In[ ]:





# In[ ]:




