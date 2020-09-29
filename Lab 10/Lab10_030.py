#Author Tushar Goyal

from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as mvn


# In[3]:


digits = load_digits()
data = scale(digits.data)
reduced_data = PCA(n_components=2).fit_transform(data)


# In[4]:


reduced_data


# In[5]:


model = KMeans(n_clusters = 10)
model.fit(reduced_data)


# In[6]:


labels = model.predict(reduced_data)


# In[7]:


model.inertia_


# In[8]:


plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:,0], reduced_data[:,1], c=model.labels_.astype(float))
centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


# In[10]:


ks = [5,8,10,12,15,17]
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model_t = KMeans(n_clusters=k)
    
    # Fit model to samples
    model_t.fit(reduced_data)
    
    # Append the inertia to the list of inertias
    inertias.append(model_t.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)


# In[27]:


gmm = GaussianMixture(n_components=5)
gmm.fit(reduced_data)


# In[30]:


gmm = GaussianMixture(n_components=10, covariance_type='full').fit(reduced_data)
prediction_gmm = gmm.predict(reduced_data)


centers = np.zeros((3,2))
for i in range(3):
    density = mvn(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(reduced_data)
    centers[i, :] = reduced_data[np.argmax(density)]

plt.figure(figsize = (10,8))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1],c=prediction_gmm ,s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6);


# In[15]:


prediction_gmm


# In[29]:


ks =[5,8,10,12,15,17]
scores = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model_t = GaussianMixture(n_components=k, covariance_type='full')
    
    # Fit model to samples
    model_t.fit(reduced_data)
    prediction_gmm = model_t.predict(reduced_data)
    # Append the inertia to the list of inertias
    scores.append(model_t.score(reduced_data))
    
# Plot ks vs inertias
plt.plot(ks, scores, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('log-likelihood')
plt.xticks(ks)


# In[35]:


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix,axis=0))/np.sum(contingency_matrix)


# In[36]:


purity_score(digits.target,prediction_gmm)


# In[ ]:




