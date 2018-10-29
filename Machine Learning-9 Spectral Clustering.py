
# coding: utf-8

# In[1]:


#Problem Statement

#In this assignment students have to compress racoon grey scale image into 5 clusters. In the end, visualize both raw 
#and compressed image and look for quality difference. The raw image is available in spicy.misc package with the name face.


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#Segmenting the picture of a raccoon face in regions

#This example uses spectral_clustering on a graph created from voxel-to-voxel difference on an image to break this image 
#into multiple partly-homogeneous regions.

#This procedure (spectral clustering on an image) is an efficient approximate solution for finding normalized graph cuts.

#There are two options to assign labels:

#with 'kmeans' spectral clustering will cluster samples in the embedding space using a kmeans algorithm
#whereas 'discrete' will iteratively search for the closest partition space to the embedding space.


# In[4]:


# Compress racoon grey scale image into 5 clusters using Sklearn Spectral Clustering Api:


# In[5]:


import numpy as np
from sklearn import cluster, datasets
from scipy import misc
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
import time


# In[6]:


#face = sp.misc.face(gray = True)
#n_clusters = 5
#np.random.seed(0)
#X = face.reshape((-1,1))
#k_Means= cluster.KMeans(n_clusters=n_clusters,n_init=4)
#k_Means.fit(X)

#plt.figure(figsize=(8, 3.5))
#plt.subplot(121)
#plt.imshow(X, cmap='gray')
#plt.axis('off')

#plt.tight_layout()
#plt.show()


# In[7]:


face = sp.misc.face(gray=True)


# In[8]:


# face

plt.imshow(face)


# In[9]:


# Resize it to 10% of the original size to speed up the processing:

face = sp.misc.imresize(face, 0.10) / 255.


# In[10]:


plt.imshow(face)


# In[11]:


#Convert the image into a graph with the value of the gradient on the edges.

graph = image.img_to_graph(face)


# In[14]:


# graph.data

graph.data.std()


# In[15]:


# Take a decreasing function of the gradient: an exponential
# The smaller beta is, the more independent the segmentation is of the
# actual image. For beta=1, the segmentation is close to a voronoi
beta = 5
eps = 1e-6
graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps


# In[16]:


# Apply spectral clustering (this step goes much faster if you have pyamg installed)

N_REGIONS = 5


# In[17]:


for assign_labels in ('kmeans', 'discretize'):
    t0 = time.time()
    labels = spectral_clustering(graph, n_clusters=N_REGIONS, assign_labels=assign_labels, random_state=1)
    
    t1 = time.time()
    labels = labels.reshape(face.shape)

    plt.figure(figsize=(5, 5))
    plt.imshow(face, cmap=plt.cm.gray)
    for l in range(N_REGIONS):
        plt.contour(labels == l, contours=1, colors=[plt.cm.nipy_spectral(l / float(N_REGIONS))])
        
    plt.xticks(())
    plt.yticks(())
    title = 'Spectral clustering: %s, %.2fs' % (assign_labels, (t1 - t0))
    print(title)
    plt.title(title)
plt.show()

