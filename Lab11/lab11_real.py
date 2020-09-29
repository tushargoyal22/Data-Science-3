from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from sklearn.cluster import AgglomerativeClustering as agnes
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import DBSCAN


#Purity score computation

from scipy.optimize import linear_sum_assignment
def purity_score(y_true, y_pred):
     # compute contingency matrix (also called confusion matrix)
     contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
     
     # Find optimal one-to-one mapping between cluster labels and true labels
     row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
     
     # Return cluster accuracy
     return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)



digits = load_digits()
data = scale(digits.data)
pca = PCA(n_components=2).fit(data)
reduced_data = PCA(n_components=2).fit_transform(data)

reduced_data1=reduced_data.copy()


plt.scatter(reduced_data[:,0],reduced_data[:,1])
plt.xlabel("X-Component")
plt.ylabel("Y-Component")
plt.show()

y_true = digits.target




print("\n\n************************Agglomerative *********************")



clt = agnes(linkage='complete', affinity='euclidean', n_clusters=10)

# Train model
model = clt.fit(reduced_data)
y_agnes = model.fit_predict(reduced_data)

labels_agg  = model.labels_

print("\nPurity Score : ",purity_score(y_true, y_agnes))

plt.xlabel("X-Component")
plt.ylabel("Y-Component")
plt.scatter(reduced_data[:,0],reduced_data[:,1],c=labels_agg, s=10, cmap='rainbow')
plt.show()



print("\n\n********************* DBSCAN *********************")



dbscan_model = DBSCAN().fit(reduced_data)

y_db = dbscan_model.fit_predict(reduced_data)

print("Purity Score : ",purity_score(y_true, y_db))


db_labels = dbscan_model.labels_

plt.xlabel("X-Component")
plt.ylabel("Y-Component")
plt.scatter(reduced_data[:,0],reduced_data[:,1],c= db_labels, s=10, cmap='rainbow')
plt.show()


ep_values=[0.05,0.5,0.95]
min_sample_values=[1,2,10,30,50]

print("-------------------- Varying Parameters --------------------")

for i in ep_values:
    for j in min_sample_values: 
        print("eps value : ",i)
        print("min samples : ",j)

        dbscan_models = DBSCAN(algorithm='auto', eps=i, metric='euclidean', min_samples=j).fit(reduced_data)
        db_labels = dbscan_models.labels_
        
#        y_db = dbscan_model.fit_predict(reduced_data)
#        print("Purity Score : ",purity_score(y_true, y_db))
        
        plt.xlabel("X-Component")
        plt.ylabel("Y-Component")
        plt.scatter(reduced_data[:,0],reduced_data[:,1],c= db_labels, s=10, cmap='rainbow')
        plt.show()
        
       


print("\n\n-----------------------------------KMeans ------------------------------ \n\n")


km = KMeans(n_clusters=10, max_iter=500)
km.fit(reduced_data)
y_kmeans = km.predict(reduced_data)
purity = purity_score(y_true, y_kmeans)
print("Purity Score for KMeans : ",purity)


labels = km.labels_

centers = km.cluster_centers_

plt.xlabel("X-Component")
plt.ylabel("Y-Component")
plt.scatter(reduced_data[:,0],reduced_data[:,1], c=labels,s=10, cmap='rainbow')
plt.scatter(centers[:,0],centers[:,1],c='black',)


plt.show()

