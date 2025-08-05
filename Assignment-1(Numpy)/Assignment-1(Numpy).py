#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

arr = np.array([1, 2, 3, 6, 4, 5])
reversed_arr = arr[::-1]
print("Reversed array", reversed_arr)


# In[2]:


array1 = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]])

flat1 = array1.flatten()
print("Flattened array using flatten():", flat1)


# In[4]:


arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[1, 2], [3, 4]])

all_equal = np.array_equal(arr1, arr2)


print("Both are equal?", all_equal)


# In[7]:


x = np.array([1, 2, 3, 4, 5, 1, 2, 1, 1, 1])
y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3])

def most_frequent(arr):
    values, counts = np.unique(arr, return_counts=True)
    maxcnt = counts.max()
    most_freq = values[counts == maxcnt]
    indices = {val: np.where(arr == val)[0] for val in most_freq}
    return most_freq, indices

print("Most frequent in array_x", most_frequent(x))
print("Most frequent in array_y", most_frequent(y))


# In[8]:


gfg = np.matrix('[4, 1, 9; 12, 3, 1; 4, 5, 6]')

total_sum = gfg.sum()

row_sum = gfg.sum(axis=1)

col_sum = gfg.sum(axis=0)

print("Sum of all elements", total_sum)
print("Sum of elements rowwise", row_sum)
print("Sum of elements colmwise", col_sum)


# In[9]:


n_array = np.array([[55, 25, 15], [30, 44, 2], [11, 45, 77]])

diag_sum = np.trace(n_array)

eigvals = np.linalg.eigvals(n_array)

eigvals, eigvecs = np.linalg.eig(n_array)

inverse = np.linalg.inv(n_array)

determinant = np.linalg.det(n_array)

print("Sum of diagonal elements:", diag_sum)
print("Eigenvalues", eigvals)
print("Eigenvectors\n", eigvecs)
print("Inverse matrix\n", inverse)
print("Determinant ", determinant)


# In[15]:



p1 = np.array([[1, 2], [2, 3]])
q1 = np.array([[4, 5], [6, 7]])

product1 = np.dot(p1, q1)
covariance1 = np.cov(p1.flatten(), q1.flatten())


# In[14]:


p2 = np.array([[1, 2], [2, 3], [4, 5]])
q2 = np.array([[4, 5, 1], [6, 7, 2]])

product2 = np.dot(p2, q2)
covariance2 = np.cov(p2.flatten(), q2.flatten())


# In[13]:


print("Product matrix1 \n", product1)
print("Covariance matrix1 \n", covariance1)
print("Product matrix2 \n", product2)
print("Covariance matrix2 \n", covariance2)


# In[16]:


x = np.array([[2, 3, 4], [3, 2, 9]])
y = np.array([[1, 5, 0], [5, 10, 3]])

inner_product = np.inner(x, y)
outer_product = np.outer(x.flatten(), y.flatten())
cartesian_product = np.array(np.meshgrid(x.flatten(), y.flatten())).T.reshape(-1, 2)

print("Inner product\n", inner_product)
print("Outer product\n", outer_product)
print("Cartesian product\n", cartesian_product)


# In[17]:


array = np.array([[1, -2, 3], [-4, 5, -6]])


abs_array = np.abs(array)

flat = array.flatten()


percentiles_flat = np.percentile(flat, [25, 50, 75])
percentiles_cols = np.percentile(array, [25, 50, 75], axis=0)
percentiles_rows = np.percentile(array, [25, 50, 75], axis=1)

mean_cols = np.mean(array, axis=0)
median_cols = np.median(array, axis=0)
std_cols = np.std(array, axis=0)


# In[18]:



mean_rows = np.mean(array, axis=1)
median_rows = np.median(array, axis=1)
std_rows = np.std(array, axis=1)


# In[19]:



mean_flat = np.mean(flat)
median_flat = np.median(flat)
std_flat = np.std(flat)


# In[20]:




print("Absvals", abs_array)
print("Percentile flat", percentiles_flat)
print("Percentiles for colms\n", percentiles_cols)
print("Percentiles for rows", percentiles_rows)

print("Mean (flat):", mean_flat, "Median (flat):", median_flat, "Std (flat):", std_flat)


# In[21]:


a = np.array([-1.8, -1.6, -0.5, 0.5, 1.6, 1.8, 3.0])

flr = np.floor(a)
ceil_val = np.ceil(a)
trunc_vals = np.trunc(a)
rnd_vals = np.round(a)

print("Floor ", flr)
print("Ceiling:", ceil_val)
print("Truncated:", trunc_vals)
print("Rounded:", rnd_vals)


# In[23]:


array = np.array([10, 52, 62, 16, 16, 54, 453])

sorted_array = np.sort(array)

sorted_indices = np.argsort(array)

smallest_4 = np.partition(array, 4)[:4]

largest_5 = np.partition(array, -5)[-5:]

print("Sorted array ", sorted_array)
print("Indices of sorted array", sorted_indices)


# In[22]:


array = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])

int_elements = array[array == array.astype(int)]

float_elements = array[array != array.astype(int)]

print("Integer elements only", int_elements)
print("Float elements only", float_elements)


# In[ ]:




