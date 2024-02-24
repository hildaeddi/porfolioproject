#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import ast
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity


# ## Reading the file/importing the file into python note

# In[54]:


acbs = pd.read_csv(r"C:\Users\seasi\Downloads\Amazon Customer Behavior Survey.csv")


# In[102]:


acbs.head()


# ## data Cleaning process .  Cheking for dublicates

# In[56]:


acbs.drop_duplicates()


# In[57]:


acbs["Gender"].str.strip()


# ## first Assumption . Replacing prefer not to say with an accepted form of gender as it is in Gender column. Nearest neighbour was assumed

# In[58]:


acbs["Gender"] = acbs["Gender"].str.replace('Prefer not to say','female')


# In[59]:


acbs


# ## viewing the complete data

# In[60]:


pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
acbs


# In[61]:


acbs.info()


# ## checking for columns with null values and returing the total 

# In[62]:


acbs.isnull().sum()


# ## removing the empty/null rolls from the data (Product_Search_Method had 2 null rolls)

# In[63]:


acbs.dropna(inplace=True)


# In[64]:


acbs.isnull().sum()


# ## converting to obj and append with the function of ast library

# In[65]:


def convert (obj):
    L= []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L    


# In[66]:


acbs.iloc[0].Customer_Reviews_Importance


# ## selecting the column from the df to be use in our model and making a new df

# In[67]:


acbs=acbs[['age','Gender','Purchase_Frequency','Purchase_Categories','Browsing_Frequency','Product_Search_Method','Search_Result_Exploration','Customer_Reviews_Importance','Review_Reliability','Review_Helpfulness','Rating_Accuracy ','Service_Appreciation','Shopping_Satisfaction']]


# In[68]:


acbs.head()


# In[69]:


acbs.head()


# In[70]:


acbs.tail(2)


# In[ ]:





# In[71]:


# Assuming 'Service_Appreciation' is a column with lists of strings
#acbs.loc[:, 'Service_Appreciation'] = acbs['Service_Appreciation'].apply(lambda lst: [item.split() for item in lst])

# If you want to flatten the resulting lists
#acbs['Service_Appreciation'] = acbs['Service_Appreciation'].explode()

# If you want to reset the index
acbs.reset_index(drop=True, inplace=True)


# In[47]:


acbs


# In[72]:


acbs['Search_Result_Exploration'] = acbs['Search_Result_Exploration'].apply(lambda lst: ''.join([char[0] for char in lst if char]))


# In[73]:


acbs


# In[74]:


acbs['Service_Appreciation'] = acbs['Service_Appreciation'].apply(lambda lst: ''.join([char[0] for char in lst if char]))
acbs


# In[75]:


acbs['groupings'] = (
    acbs['Purchase_Categories'].astype(str) +
    acbs['Purchase_Frequency'].astype(str) +
    acbs['Browsing_Frequency'].astype(str) +
    acbs['Product_Search_Method'].astype(str) +
    acbs['Search_Result_Exploration'].apply(lambda lst: ''.join([char[0] for char in lst if char])).astype(str) +
    acbs['Customer_Reviews_Importance'].astype(str) +
    acbs['Review_Reliability'].astype(str) +
    acbs['Review_Helpfulness'].astype(str) +
    acbs['Rating_Accuracy '].astype(str) +
    acbs['Service_Appreciation'].astype(str) +
    acbs['Shopping_Satisfaction'].astype(str)
)


# In[ ]:





# In[76]:


new_df = acbs[['Gender','age','groupings']]
new_df


# In[77]:


acbs.head(2)
#['Product_Category'] = df['Product_Category'].apply(lambda x: x.strip('[]'))


# In[78]:


#new_df['Purchase_Categories'] = new_df['Purchase_Categories'].apply(lambda x: x.strip('[]').strip())

#new_df['Purchase_Categories'] = new_df['Purchase_Categories'].apply(lambda x: x.strip('[]'))
#new_df['Purchase_Categories'] = new_df['Purchase_Categories'].apply(lambda x: str(x).strip('[]').strip())
new_df['groupings'] = new_df['groupings'].apply(lambda x: str(x).strip('[]').strip())
#new_df['Purchase_Categories'] = new_df['Purchase_Categories'].apply(lambda x: str(x).replace("'", ""))
new_df['groupings'] = new_df['groupings'].apply(lambda x: str(x).replace("'", ""))

new_df.head(2)


# In[79]:


new_df['groupings'] = new_df['groupings'].apply(lambda x: x.strip('[]').strip())
new_df.head(3)


# In[80]:


new_df.shape


# In[81]:


# Assuming 'column_name' is the name of the column you want to access
groupings = new_df['groupings']
groupings.head(4)


# In[82]:


# Assuming 0 is the index of the column you want to access
groupings = new_df.iloc[:, 0]


# In[83]:


# Sample DataFrame
#data = {'groupings': ["23['Few', 'times', 'a', 'month']", "1['Once', 'a', 'month']['Few', 'times', 'a', 'week']",
                   #   "2['Few', 'times', 'a', 'month']", "3['Once', 'a', 'month']['Few', 'times', 'a', 'week']"]}

#new_df = pd.DataFrame(data)

# Applying the code to clean the 'groupings' column
new_df['groupings'] = new_df['groupings'].apply(lambda x: ', '.join([word.strip("[]''") for word in x.split(',')]))

# Additional step to remove single quotes around each word
new_df['groupings'] = new_df['groupings'].apply(lambda x: ' '.join([word.strip("'") for word in x.split()]))

# Display the updated DataFrame
print(groupings)


# In[84]:


new_df['Gender']= new_df['Gender'].str.lower()
new_df['groupings']= new_df['groupings'].str.lower()

new_df.head(2)


# In[85]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=4000,stop_words= 'english')


# In[86]:


cv.fit_transform(new_df['groupings']).toarray().shape


# In[87]:


vectors = cv.fit_transform(new_df['groupings']).toarray()
vectors[1]


# In[88]:


len(cv.get_feature_names_out())


# ## import nltk
# ### from nltk.stem.porter import PorterStemmer
# ### ps = PorterStemmer()

# In[89]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ". join(y)


# In[90]:


new_df['groupings']=new_df['groupings'].apply(stem)


#  ##  From sklearn import metrics
# # ## from sklearn.metrics.pairwise import cosine_similarity

# In[91]:


cosine_similarity(vectors)


# In[92]:


cosine_similarity(vectors).shape


# In[93]:


similarity = cosine_similarity(vectors)


# In[94]:


sorted(list(enumerate(similarity[0])),reverse = True,key=lambda x:x[1])[1:6]


# ## Creating the recommendation system

# In[95]:


#def recommend(acbs):
 #   acbs_index =new_df[new_df['Gender']==acbs].index[0]
  #  distances=similarity[acbs_index]
   # acbs_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    #for i in movies_list:
     #   print(new_df.iloc[i[0]].title)
        
#def recommend(acbs):
 #   try:
  #      acbs_index = new_df[new_df['female'] == acbs].index[0]
   # except IndexError:
    #    return "No data found for the specified gender."

    #distances = similarity[acbs_index]
    #acbs_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    #return acbs_list
    
def recommend(acbs):
    try:
        acbs_index = new_df[new_df['age'] == acbs].index[0]
    except IndexError:
        return "No data found for the specified gender."

    distances = similarity[acbs_index]
    acbs_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return acbs_list



# In[96]:


recommend(25)


# In[97]:


recommend(23)


# In[98]:


recommend(24)


# In[99]:


def recommend(acbs):
    try:
        acbs_index = new_df[new_df['Gender'] == acbs].index[0]
    except IndexError:
        return "No data found for the specified gender."

    distances = similarity[acbs_index]
    acbs_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return acbs_list


# In[100]:


recommend('female')


# In[101]:


recommend('male')


# In[ ]:




