#!/usr/bin/env python
# coding: utf-8

# In[75]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_excel('german_credit_data (1).xlsx')


# In[76]:


data['Age'].value_counts()


# In[ ]:





# In[ ]:





# In[4]:


data['Risk'].value_counts()


# In[5]:


data.isnull().sum()


# In[6]:


y = data['Risk']
X = data.drop(['Risk'],axis=1)


# In[7]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X,y,stratify = y,test_size = 0.3,random_state = 42)


# In[8]:


data_train = pd.concat((X_train, y_train),
                       axis = 1)


# In[9]:


sns.countplot(data = data_train,
              x = 'Housing',
              hue = 'Risk')


# In[10]:


# Create a function for binning the numerical predictor
def create_binning(data, predictor_label, num_of_bins):
    """
    Function for binning numerical predictor.

    Parameters
    ----------
    data : array like
      The name of dataset.

    predictor_label : object
      The label of predictor variable.

    num_of_bins : integer
      The number of bins.


    Return
    ------
    data : array like
      The name of transformed dataset.

    """
    # Create a new column containing the binned predictor
    data[predictor_label + "_bin"] = pd.qcut(data[predictor_label],
                                             q = num_of_bins)

    return data


# In[11]:


data.info()


# In[12]:


data.columns


# In[13]:


# Define data with numerical predictors
num_columns = ['Age','Credit amount','Duration']


# In[14]:


# Define data with categorical predictors
cat_columns = ['Sex','Job','Housing','Saving accounts','Checking account','Purpose']


# In[15]:


import seaborn as sns


# In[16]:


data[['Age','Credit amount','Duration']].corr(method = 'pearson')


# In[17]:


for column in num_columns:
  data_train_binned = create_binning(data = data_train,
                                     predictor_label = column,
                                     num_of_bins = 4)


# In[18]:


data_train_binned.T


# In[19]:


# Check for missing values
data_train_binned.isna().sum()


# In[20]:


cat_columns = ['Sex','Job','Housing','Saving accounts','Checking account','Purpose']


# In[21]:


# Define columns with missing values
missing_columns = ['Saving accounts','Checking account']


# In[ ]:





# In[22]:


data[['Sex','Job','Housing','Saving accounts','Checking account','Purpose']] = data[['Sex','Job','Housing','Saving accounts','Checking account','Purpose']].astype('category')


# In[23]:


for column in missing_columns:
    # Convert the column to 'category' data type if it's not already
    data_train_binned[column] = data_train_binned[column].astype('category')

    # Add category 'Missing' to replace the missing values
    data_train_binned[column] = data_train_binned[column].cat.add_categories('Missing')

    # Replace missing values with category 'Missing'
    data_train_binned[column].fillna(value='Missing', inplace=True)


# In[24]:


# Sanity check
data_train_binned.isna().sum()


# In[25]:


data_train_binned['Checking account'].value_counts()


# In[26]:


# Define the initial empty list
crosstab_num = []

for column in num_columns:

  # Create a contingency table
  crosstab = pd.crosstab(data_train_binned[column + "_bin"],
                         data_train_binned['Risk'],
                         margins = True)

  # Append to the list
  crosstab_num.append(crosstab)


# In[27]:


# Define the initial empty list
crosstab_cat = []

for column in cat_columns:

  # Create a contingency table
  crosstab = pd.crosstab(data_train_binned[column],
                         data_train_binned['Risk'],
                         margins = True)

  # Append to the list
  crosstab_cat.append(crosstab)


# In[28]:


# Put all two in a crosstab_list
crosstab_list = crosstab_num + crosstab_cat

crosstab_list


# In[29]:


# Define the initial list for WOE
WOE_list = []

# Define the initial list for IV
IV_list = []

# Create the initial table for IV
IV_table = pd.DataFrame({'Characteristic': [],
                         'Information Value' : []})

# Perform the algorithm for all crosstab
for crosstab in crosstab_list:

  # Calculate % Good
  crosstab['p_good'] = crosstab['good']/crosstab['good']['All']

  # Calculate % Bad
  crosstab['p_bad'] = crosstab['bad']/crosstab['bad']['All']

  # Calculate the WOE
  crosstab['WOE'] = np.log(crosstab['p_good']/crosstab['p_bad'])

  # Calculate the contribution value for IV
  crosstab['contribution'] = (crosstab['p_good']-crosstab['p_bad'])*crosstab['WOE']

  # Calculate the IV
  IV = crosstab['contribution'][:-1].sum()

  add_IV = {'Characteristic': crosstab.index.name,
            'Information Value': IV}

  WOE_list.append(crosstab)
  IV_list.append(add_IV)


# In[30]:


WOE_list


# In[31]:


# Create initial table to summarize the WOE values
WOE_table = pd.DataFrame({'Characteristic': [],
                          'Attribute': [],
                          'WOE': []})

for i in range(len(crosstab_list)):

  # Define crosstab and reset index
  crosstab = crosstab_list[i].reset_index()

  # Save the characteristic name
  char_name = crosstab.columns[0]

  # Only use two columns (Attribute name and its WOE value)
  # Drop the last row (average/total WOE)
  crosstab = crosstab.iloc[:-1, [0,-2]]
  crosstab.columns = ['Attribute', 'WOE']

  # Add the characteristic name in a column
  crosstab['Characteristic'] = char_name

  WOE_table = pd.concat((WOE_table, crosstab),
                        axis = 0)

  # Reorder the column
  WOE_table.columns = ['Characteristic',
                       'Attribute',
                       'WOE']

WOE_table


# In[32]:


# Put all IV in the table
IV_table = pd.DataFrame(IV_list)
IV_table


# In[33]:


# Define the predictive power of each characteristic
strength = []

# Assign the rule of thumb regarding IV
for iv in IV_table['Information Value']:
  if iv < 0.02:
    strength.append('Unpredictive')
  elif iv >= 0.02 and iv < 0.1:
    strength.append('Weak')
  elif iv >= 0.1 and iv < 0.3:
    strength.append('Medium')
  else:
    strength.append('Strong')

# Assign the strength to each characteristic
IV_table = IV_table.assign(Strength = strength)

# Sort the table by the IV values
IV_table.sort_values(by='Information Value', ascending=False)


# In[34]:


# Create a funtion to plot the WOE
def plot_WOE(crosstab):
  """
  Function to plot the WOE trend.

  Parameters
  ----------
  crosstab : DataFrame
    The cross tabulation of the characteristic.

  """
  # Define the plot size
  plt.figure(figsize = (8,4))

  # Plot the WOE
  sns.pointplot(x = crosstab.T.columns,
                y = 'WOE',
                data = crosstab,
                markers = 'o',
                linestyles = '--',
                color = 'blue')

  # Rotate the label of x-axis
  plt.xticks(rotation = 20)


# In[35]:


crosstab_list


# In[36]:


# Define the crosstab
crosstab_age_bin = crosstab_list[0]

crosstab_age_bin


# In[37]:


# Plot the WOE
plot_WOE(crosstab_age_bin)


# In[38]:


# Define the crosstab
crosstab_credit_amount_bin = crosstab_list[1]

crosstab_credit_amount_bin


# In[39]:


plot_WOE(crosstab_credit_amount_bin)


# In[40]:


# Define the crosstab
crosstab_duration_bin = crosstab_list[2]

crosstab_duration_bin


# In[41]:


plot_WOE(crosstab_duration_bin)


# In[42]:


crosstab_sex = crosstab_list[3]

crosstab_sex


# In[43]:


plot_WOE(crosstab_sex)


# Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)

# In[44]:


crosstab_job = crosstab_list[4]

crosstab_job


# In[45]:


plot_WOE(crosstab_job)


# In[46]:


crosstab_housing= crosstab_list[5]

crosstab_housing


# In[47]:


plot_WOE(crosstab_housing)


# In[48]:


crosstab_saving_accounts= crosstab_list[6]

crosstab_saving_accounts


# 

# In[49]:


plot_WOE(crosstab_saving_accounts)


# In[50]:


crosstab_checking_account= crosstab_list[7]

crosstab_checking_account


# In[51]:


plot_WOE(crosstab_checking_account)


# In[52]:


crosstab_purpose= crosstab_list[8]

crosstab_purpose


# In[53]:


plot_WOE(crosstab_purpose)


# In[54]:


# Function to generate the WOE mapping dictionary
def get_woe_map_dict(WOE_table):

    # Initialize the dictionary
    WOE_map_dict = {}
    WOE_map_dict['Missing'] = {}

    unique_char = set(WOE_table['Characteristic'])
    for char in unique_char:
        # Get the Attribute & WOE info for each characteristics
        current_data = (WOE_table
                            [WOE_table['Characteristic']==char]     # Filter based on characteristic
                            [['Attribute', 'WOE']])                 # Then select the attribute & WOE

        # Get the mapping
        WOE_map_dict[char] = {}
        for idx in current_data.index:
            attribute = current_data.loc[idx, 'Attribute']
            woe = current_data.loc[idx, 'WOE']

            if attribute == 'Missing':
                WOE_map_dict['Missing'][char] = woe
            else:
                WOE_map_dict[char][attribute] = woe
                WOE_map_dict['Missing'][char] = np.nan

    # Validate data
    print('Number of key : ', len(WOE_map_dict.keys()))

    return WOE_map_dict


# In[55]:


# Function to replace the raw data in the train set with WOE values
def transform_woe(raw_data, WOE_dict, num_cols):

    woe_data = raw_data.copy()

    # Map the raw data
    for col in woe_data.columns:
        if col in num_cols:
            map_col = col + '_bin'
        else:
            map_col = col

        woe_data[col] = woe_data[col].map(WOE_map_dict[map_col])

    # Map the raw data if there is a missing value or out of range value
    for col in woe_data.columns:
        if col in num_cols:
            map_col = col + '_bin'
        else:
            map_col = col

        woe_data[col] = woe_data[col].fillna(value=WOE_map_dict['Missing'][map_col])

    return woe_data


# In[56]:


# Generate the WOE map dictionary
WOE_map_dict = get_woe_map_dict(WOE_table = WOE_table)
WOE_map_dict


# In[57]:


# Transform the X_train
woe_train = transform_woe(raw_data = X_train,
                          WOE_dict = WOE_map_dict,
                          num_cols = num_columns)

woe_train


# In[58]:


# Transform the X_test
woe_test = transform_woe(raw_data = X_test,
                         WOE_dict = WOE_map_dict,
                         num_cols = num_columns)

woe_test


# In[59]:


# Rename the raw X_train for the future
raw_train = X_train
raw_train


# In[60]:


raw_test=X_test
raw_test


# In[61]:


# Define X_train
X_train = woe_train.to_numpy()
X_train


# In[62]:


# Define X_train
X_test = woe_test.to_numpy()
X_test


# In[63]:


# Check y_train
y_train = y_train.to_numpy()
y_train


# In[64]:


y_train[y_train == 'bad'] = 1
y_train[y_train == 'good'] = 0


# In[65]:


y_train = y_train.astype('i')
y_train = np.int64(y_train)
y_train


# In[66]:


# Check y_train
y_train


# In[67]:


y_test[y_test == 'bad'] = 1
y_test[y_test == 'good'] = 0


# In[68]:


y_test = y_test.astype('i')
y_test = np.int64(y_test)
y_test


# In[69]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


# In[70]:


logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)


# In[77]:


# Step 3: Make predictions on the test set
y_pred = logreg_model.predict(X_test)


# In[72]:


# Confusion matrix
conf_matrix = confusion_matrix(y_train, y_pred)
print("Confusion Matrix:\n", conf_matrix)


# In[73]:


# Classification report
classification_rep = classification_report(y_train, y_pred)
print("Classification Report:\n", classification_rep)


# In[74]:


# ROC AUC score (if applicable for binary classification)
roc_auc = roc_auc_score(y_train, y_pred)
print(f"ROC AUC: {roc_auc:.4f}")


# In[ ]:




