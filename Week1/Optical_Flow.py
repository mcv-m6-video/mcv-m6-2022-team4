#!/usr/bin/env python
# coding: utf-8

# In[48]:


import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import flow_vis


# In[2]:


img45 = cv2.cvtColor(cv2.imread(os.path.join("results", "LKflow_000045_10.png"), cv2.IMREAD_UNCHANGED).astype(np.uint16), cv2.COLOR_BGR2RGB)
img157 = cv2.cvtColor(cv2.imread(os.path.join("results", "LKflow_000157_10.png"), cv2.IMREAD_UNCHANGED).astype(np.uint16), cv2.COLOR_BGR2RGB)
img45_GT = cv2.cvtColor(cv2.imread(os.path.join("results", "000045_10.png"), cv2.IMREAD_UNCHANGED).astype(np.uint16), cv2.COLOR_BGR2RGB)
img157_GT = cv2.cvtColor(cv2.imread(os.path.join("results", "000157_10.png"), cv2.IMREAD_UNCHANGED).astype(np.uint16), cv2.COLOR_BGR2RGB)


# In[64]:


def show_field(flow, gray, step=30, scale=0.5):
    
    plt.figure(figsize=(16,8))
    plt.imshow(gray, cmap='gray')
    
    U = flow[:, :, 0]
    V = flow[:, :, 1]
    H = np.hypot(U, V)

    (h, w) = flow.shape[0:2]
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))

    x = x[::step, ::step]
    y = y[::step, ::step]
    U = U[::step, ::step]
    V = V[::step, ::step]
    H = H[::step, ::step]

    plt.quiver(x, y, U, V, H, scale_units='xy', angles='xy', scale=scale)
    
    plt.axis('off')
    plt.show()


# In[3]:


def transform_annotation(img):
    
    flow_u = (img[:,:,0].astype(float) - 2. ** 15) / 64.0
    flow_v = (img[:,:,1].astype(float) - 2. ** 15) / 64.0
    valid  = img[:,:,2].astype(bool)
    
    return flow_u, flow_v, valid


# In[88]:


GT_u, GT_v, GT_valid = transform_annotation(img157_GT)
u, v, r_valid = transform_annotation(img157)
motion_vectors_dist, msen, pepn = get_metrics(GT_u, GT_v, u, v, GT_valid, 3)
print("MSEN: {}, PEPN: {}".format(msen, pepn))


fig = plt.figure(figsize=(12,9))
rge = motion_vectors_dist.max() - motion_vectors_dist.min()
plt.title("Error (distance) for img157")
im = plt.imshow(motion_vectors_dist, cmap="hot")
cbar_ax = fig.add_axes([0.93, 0.35, 0.02, 0.3])
fig.colorbar(im, cax = cbar_ax)
plt.show()


ths = list(range(55)) #int(motion_vectors_dist[GT_valid != 0].max())))
plt.figure(figsize=(12,8))
plt.title("Errors distribution (in pixels) for img157")
sns.histplot(motion_vectors_dist[GT_valid != 0], stat="density", bins=range(25), color="red")
plt.xlim(0, 50)
plt.show()

    


# In[69]:


gray = cv2.cvtColor(cv2.imread(os.path.join("results", "colored_000157_10.png")), cv2.COLOR_BGR2GRAY)
show_field(np.dstack((GT_u, GT_v, GT_valid)),gray,step=15,scale=.3)

flow_color = flow_vis.flow_to_color(np.dstack((GT_u, GT_v)), convert_to_bgr=False)
Image.fromarray(flow_color)


# In[90]:


GT_u, GT_v, GT_valid = transform_annotation(img45_GT)
u, v, _ = transform_annotation(img45)
motion_vectors_dist, msen, pepn = get_metrics(GT_u, GT_v, u, v, GT_valid, 3)
print("MSEN: {}, PEPN: {}".format(msen, pepn))


fig = plt.figure(figsize=(12,9))
rge = motion_vectors_dist.max() - motion_vectors_dist.min()
plt.title("Error (distance) for img45")
im = plt.imshow(motion_vectors_dist, cmap="hot")
cbar_ax = fig.add_axes([0.93, 0.35, 0.02, 0.3])
fig.colorbar(im, cax = cbar_ax)
plt.show()


ths = list(range(55)) #int(motion_vectors_dist[GT_valid != 0].max())))
plt.figure(figsize=(12,8))
plt.title("Errors distribution (in pixels) for img45")
sns.histplot(motion_vectors_dist[GT_valid != 0], stat="density", bins=range(50), color="orange")
plt.xlim(0, 50)
plt.show()

    


# In[72]:


flow_color = flow_vis.flow_to_color(np.dstack((GT_u, GT_v)), convert_to_bgr=False)
Image.fromarray(flow_color)

gray = cv2.cvtColor(cv2.imread(os.path.join("results", "colored_000045_10.png")), cv2.COLOR_BGR2GRAY)
show_field(np.dstack((GT_u, GT_v, GT_valid)),gray,step=15,scale=.3)

