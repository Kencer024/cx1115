#!/usr/bin/env python
# coding: utf-8

# ---
# 
# ### Essential Libraries
# 
# Let us begin by importing the essential Python Libraries.
# 
# > NumPy : Library for Numeric Computations in Python  
# > Pandas : Library for Data Acquisition and Preparation  
# > Matplotlib : Low-level library for Data Visualization  
# > Seaborn : Higher-level library for Data Visualization  

# In[1]:


# Basic Libraries
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt # we only need pyplot
sb.set() # set the default Seaborn style for graphics


# In[2]:


trndata = pd.read_csv('train.csv')
trndata.head()


# In[3]:


print("Data type : ", type(trndata))
print("Data dims : ", trndata.shape)


# In[4]:


print(trndata.dtypes)


# In[5]:


# Information about the Variables
trndata.info()


# In[6]:


houseNumData = pd.DataFrame(trndata[['LotArea', 'GrLivArea', 'TotalBsmtSF', 'GarageArea', 'SalePrice']])


# In[8]:


houseNumData.head(20)


# ## Description of the house data

# In[9]:


houseNumData.describe()


# In[18]:


SP = pd.DataFrame(trndata['SalePrice'])
SP.head(10)


# In[17]:


SP.info()


# ## Graph of SalePrice

# In[13]:


f = plt.figure(figsize=(16, 10))
sb.kdeplot(data = SP)


# From the data, we can see that the graph is positively skewed 

# In[22]:


skewSP = SP.skew(axis=0)
print(skewSP)


#  SalePrice has a skewness value of +1.882876

# In[23]:


skewHND = houseNumData.skew(axis =0)
print(skewHND)


# GarageArea is the least skewed with only 0.17 value and LotArea is the most skewed with 12.2 

# In[36]:


# Draw the distributions of all variables
f, axes = plt.subplots(5, 1, figsize=(18, 30))

count = 0
for var in houseNumData[0:1]:
    #f = plt.figure(figsize=(16, 10))
    sb.kdeplot(data = houseNumData[var],ax=axes[count])
    count += 1
    


# In[37]:


# Calculate the complete  correlation matrix
houseNumData.corr()


# In[38]:


# Heatmap of the Correlation Matrix
f = plt.figure(figsize=(12, 12))
sb.heatmap(houseNumData.corr(), vmin = -1, vmax = 1, annot = True, fmt = ".2f")


# GarageArea with LotArea has the least correlation while GrLivArea and SalePrice has the most correlation. 
# In general, most variables are relatively not correlated with a value of less than 0.5 

# ## Concating the most correlated variables(SP and GrLivArea)

# In[39]:


GLA = pd.DataFrame(trndata['GrLivArea'])
GLA.head(10)


# In[43]:


# Create a joint dataframe by concatenating the two variables
jointDF = pd.concat([GLA, SP], axis = 1).reindex(GLA.index)
jointDF


# In[44]:


# Draw jointplot of the two variables in the joined dataframe
sb.jointplot(data = jointDF, x = "GrLivArea", y = "SalePrice", height = 12)


# From the graph,it is easy to see a clear correlation, especially at the lower price areas that as GLA increases, SP increases too. However, around the 2.5k mark on GLA, the points starts to be more scattered and unpredictable hence concluding that from 0 to 2.5k GLA, it is accurate for prediction of SP however after that, it is not accurate

# In[ ]:





# In[ ]:





# ## Prob2

# In[45]:


houseCatData = pd.DataFrame(trndata[['MSSubClass', 'Neighborhood', 'BldgType', 'OverallQual']])
houseCatData.head(10)


# In[47]:


# MSSubClass in the Dataset
print("Number of MSSubClass :", len(trndata["MSSubClass"].unique()))

# count in each MSSubClass
print(trndata["MSSubClass"].value_counts())
sb.catplot(y = "MSSubClass", data = trndata, kind = "count")


# From the plot, its easy to identify that the MSSubClass for 20 and 60 dominates the market with the highest count

# #### use.astype('category') to change int to cat

# In[55]:


MSSC = pd.DataFrame(houseCatData['MSSubClass']) 
MSSC.astype('category').dtypes


# In[53]:


sb.catplot(y = "MSSubClass", data = MSSC, kind = "count")


# In[61]:


# Neighborhood data
NB = pd.DataFrame(houseCatData['Neighborhood']) 
NB.astype('category').dtypes
sb.catplot(y = "Neighborhood", data = NB, kind = "count")


# In[62]:


OvQ = pd.DataFrame(houseCatData['OverallQual'])
OvQ.astype('category').dtypes


# In[70]:



dualtype_data = houseCatData[houseCatData["OverallQual"].isnull() == False]
print("Overall Qual and MSSubClass :", len(dualtype_data))


# Distribution of the MSSC & OvQ
f = plt.figure(figsize=(20, 20))
sb.heatmap(dualtype_data.groupby(['MSSubClass', 'OverallQual']).size().unstack(), 
           linewidths = 1, annot = True, annot_kws = {"size": 18}, cmap = "BuGn")


# In[71]:



# Distribution of NB & OvQ
f = plt.figure(figsize=(20, 20))
sb.heatmap(dualtype_data.groupby(['Neighborhood', 'OverallQual']).size().unstack(), 
           linewidths = 1, annot = True, annot_kws = {"size": 18}, cmap = "BuGn")


# In[72]:



# Distribution of the Two Types
f = plt.figure(figsize=(20, 20))
sb.heatmap(dualtype_data.groupby(['BldgType', 'OverallQual']).size().unstack(), 
           linewidths = 1, annot = True, annot_kws = {"size": 18}, cmap = "BuGn")


# In[75]:


#boxplot = trndata.boxplot(row = SP, column =houseCatData)
#ax = sb.boxplot(x=houseCatData, y="SalePrice", data=trndata)
# Draw the Boxplots of all variables
f = plt.figure(figsize=(16, 8))
sb.boxplot(data = houseCatData, orient = "h")


# In[ ]:




