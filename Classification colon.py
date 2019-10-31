#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load dataset
import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv(r'D:\GitHub\TA_Classification_Microarray\dataset\colon\colonTumor.data', header=-1)


# In[3]:


x = data.iloc[:, :-1] #mengambil semua row dan kolom kecuali kolom terakhir
y = data.iloc[:, -1:]#mengambil semua baris namun hanya kolom terakhir
# scaler = StandardScaler()
# X_std = scaler.fit_transform(X_std,x)
print(x)


# In[4]:


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# In[5]:


x= NormalizeData(x)
np.ravel(y)
print(x)


# In[6]:


from sys import stdout
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


# In[7]:


# Split data to train and test on 50-50 ratio
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=None)


# In[8]:



pls = PLSRegression(n_components=20)
pls.fit(X_train, X_test)
#X_pls = pls.fit_transform(X_train, X_test)
x2 =pls.transform(x)
y_c = pls.predict(x)


# In[64]:


#two_arrays = X_pls
#datapls = np.hstack(two_arrays)
#np.savetxt('lungcancerpls111.csv', datapls, delimiter=',')


# In[65]:


#import pandas as pd
#data2 = pd.read_csv(r'lungcancerpls111.csv', header=-1)


# In[9]:


#datapls= pd.DataFrame(datapls)
y_c= pd.DataFrame(y_c)
#x2= pd.DataFrame(x2)


# In[ ]:





# In[10]:


from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


# In[75]:


from sklearn.model_selection import cross_val_score
print(y_c)
i=2
#y_c.shape


# In[76]:


#create a new KNN model
while i<7:
    knn = KNeighborsClassifier(n_neighbors= i)
    cv_scores = cross_val_score(knn, x2,y , cv= i)
    a = cv_scores
    print(cv_scores)
    print("cv_scores mean:{}".format(np.mean(cv_scores)))
    i+=1
#knn.fit(y_c,)
#y_pred= knn.predict(X_test)


# In[99]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
j=2


# In[100]:


while j<7:
    classifier = SVC(kernel='linear', C=10, gamma=1)
    classifier.fit(X_train,y_train)
    cv_scores = cross_val_score(classifier , x2,y , cv=j)
    print(cv_scores)
    j+=1


# In[ ]:





# In[ ]:





# In[ ]:




