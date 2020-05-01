# recommendation-system
Build a recommendation engine solution, i.e. a solution that will automatically recommend a product to a customer based on their purchase history.
# Import modules

%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
import time
import turicreate as tc
from sklearn.model_selection import train_test_split

import sys
sys.path.append("..")

#Load data
data = pd.read_csv('AHG.csv')

# After loading the dataset, we should look at the content of each columns.

# Looking at the  file
print("\nDataFrame:")
print("shape : ", data.shape)
print("Unique Vales :", data.nunique())
print(data.head())

Create Dummy
Dummy for marking whether a customer bought that item or not. If one buys an item, then purchase_dummy are marked as 1

def create_data_dummy(data):
    data_dummy = data.copy()
    data_dummy['purchase_dummy'] = 1
    return data_dummy
data_dummy = create_data_dummy(data)

# Normalize item values across Customers

df_matrix = pd.pivot_table(data, values='SalesOrderLineNumber', 
                           index='CustomerID', columns='ProductID')

df_matrix

df_matrix_norm = (df_matrix-df_matrix.min())/(df_matrix.max()-df_matrix.min())

df_matrix_norm

# create a table for input to the modeling  
d = df_matrix_norm.reset_index() 
d.index.names = ['scaled_purchase_freq'] 
data_norm = pd.melt(d, id_vars=['CustomerID'], value_name='scaled_purchase_freq').dropna()
print(data_norm.shape)
data_norm.head()

The above steps can be combined to a function defined below:

def normalize_data(data):
    df_matrix = pd.pivot_table(data, values='SalesOrderLineNumber', index='CustomerID', columns='ProductID')
    df_matrix_norm = (df_matrix-df_matrix.min())/(df_matrix.max()-df_matrix.min())
    d = df_matrix_norm.reset_index()
    d.index.names = ['scaled_purchase_freq']
    return pd.melt(d, id_vars=['CustomerID'], value_name='scaled_purchase_freq').dropna()
    
    In this step, we have normalized their purchase history, from 0–1 (with 1 being the most number of purchase for an item and 0 being 0 purchase count for that item).

Split train and test set

# We use 80:20 ratio for our train-test set size.
#Our training portion will be used to develop a predictive model, while the other to evaluate the model’s performance.

def split_data(data):
    '''
    Splits dataset into training and test set.
    
    Args:
        data (pandas.DataFrame)
        
    Returns
        train_data (tc.SFrame)
        test_data (tc.SFrame)
    '''
    train, test = train_test_split(data, test_size = .2)
    train_data = tc.SFrame(train)
    test_data = tc.SFrame(test)
    return train_data, test_data
    
    #Now that we have three datasets with SalesOrderLineNumber, purchase dummy, and scaled purchase counts, 
#we would like to split each for modeling.

train_data, test_data = split_data(data)
train_data_dummy, test_data_dummy = split_data(data_dummy)
train_data_norm, test_data_norm = split_data(data_norm)

# Define Models using Turicreate library

# constant variables to define field names include:
user_id = 'CustomerID'
item_id = 'ProductID'
users_to_recommend = list(data[user_id])
n_rec = 10 # number of items to recommend
n_display = 30 # to display the first few rows in an output dataset

def model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display):
    if name == 'popularity':
        model = tc.popularity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target)
    elif name == 'cosine':
        model = tc.item_similarity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target, 
                                                    similarity_type='cosine')
    elif name == 'pearson':
        model = tc.item_similarity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target, 
                                                    similarity_type='pearson')
        
    recom = model.recommend(users=users_to_recommend, k=n_rec)
    recom.print_rows(n_display)
    return model
    
    # Using Popularity Model as Baseline
    
    #Using SalesOrderLineNumber

name = 'popularity'
target = 'SalesOrderLineNumber'
popularity = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

#Using purchase_dummy

name = 'popularity'
target = 'purchase_dummy'
pop_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

#Using scaled_purchased_freq

name = 'popularity'
target = 'scaled_purchase_freq'
pop_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

# Using Cosine similarity

#Using SalesOrderLineNumber

name = 'cosine'
target = 'SalesOrderLineNumber'
cos = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

#Using purchase dummy

name = 'cosine'
target = 'purchase_dummy'
cos_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

#Using scaled_purchase_freq

name = 'cosine' 
target = 'scaled_purchase_freq' 
cos_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

# Pearson similarity

#Using SalesOrderLineNumber

name = 'pearson'
target = 'SalesOrderLineNumber'
pear = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

#Using purchase dummy

name = 'pearson'
target = 'purchase_dummy'
pear_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

#Using scaled_purchase_freq

name = 'pearson'
target = 'scaled_purchase_freq'
pear_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

# Model Evaluation
We will use RMSE and Precision-recall to evalute this task.

#Let’s first create initial callable variables for model evaluation:

models_w_counts = [popularity, cos, pear]
models_w_dummy = [pop_dummy, cos_dummy, pear_dummy]
models_w_norm = [pop_norm, cos_norm, pear_norm]
names_w_counts = ['Popularity Model on SalesOrderLineNumber', 'Cosine Similarity on SalesOrderLineNumber', 'Pearson Similarity on SalesOrderLineNumber']
names_w_dummy = ['Popularity Model on Purchase Dummy', 'Cosine Similarity on Purchase Dummy', 'Pearson Similarity on Purchase Dummy']
names_w_norm = ['Popularity Model on Scaled Purchase Counts', 'Cosine Similarity on Scaled Purchase Counts', 'Pearson Similarity on Scaled Purchase Counts']

#Lets compare all the models we have built based on RMSE and precision-recall characteristics:

eval_counts = tc.recommender.util.compare_models(test_data, models_w_counts, model_names=names_w_counts)
eval_dummy = tc.recommender.util.compare_models(test_data_dummy, models_w_dummy, model_names=names_w_dummy)
eval_norm = tc.recommender.util.compare_models(test_data_norm, models_w_norm, model_names=names_w_norm)

Evaluation Summary

Popularity v. Collaborative Filtering:
We can see that the collaborative filtering algorithms work better than popularity model for SalesOrderLineNumber. Indeed, popularity model doesn’t give any personalizations as it only gives the same list of recommended items to every user.

Precision and recall:
Looking at the summary above, we see that the precision and recall for SalesOrderLineNumber > Purchase Dummy > Normalized SalesOrderLineNumber. However, because the recommendation scores for the normalized purchase data is zero and constant, we choose the dummy. In fact, the RMSE isn’t much different between models on the dummy and those on the normalized data.

RMSE:
Since RMSE is higher using pearson distance than cosine, we would choose model the smaller mean squared errors, which in this case would be cosine.

Final Output

final_model = tc.item_similarity_recommender.create(tc.SFrame(data_norm), 
                                            user_id=user_id, 
                                            item_id=item_id, 
                                            target= target,
                                                    similarity_type ='cosine')

recom = final_model.recommend(users=users_to_recommend, k=n_rec)
recom.print_rows(n_display)

# CSV output file

#Here we want to manipulate our result to a csv output. Let’s see what we have:

df_rec = recom.to_dataframe()
print(df_rec.shape)
df_rec.head()

import os
os.getcwd() 

#Let’s define a function to create a desired output:
def create_output(model, users_to_recommend, n_rec, print_csv=True):
    recomendation = model.recommend(users=users_to_recommend, k=n_rec)
    df_rec = recomendation.to_dataframe()
    df_rec['recommendedProducts'] = df_rec.groupby([user_id])[item_id] \
        .transform(lambda x: '|'.join(x.astype(str)))
    df_output = df_rec[['CustomerID', 'recommendedProducts']].drop_duplicates() \
        .sort_values('CustomerID').set_index('CustomerID')
    if print_csv:
        df_output.to_csv('C:\\Users\\Hp\output\Recommendation.csv')
        print("An output file can be found in 'output' folder with name 'Recommendation.csv'")
    return df_output
    
    #Lets print the output below and setprint_csv to true, this way we could literally print out our output file in csv,

df_output = create_output(pear_norm, users_to_recommend, n_rec, print_csv=True)
print(df_output.shape)
df_output.head()

# Customer Recommendation Function

def customer_recomendation(CustomerID):
    if CustomerID not in df_output.index:
        print('Customer not found.')
        return CustomerID
    return df_output.loc[CustomerID]
    
    #let gets product recommended to specifice custimer
customer_recomendation(11000)

# Conclusion
Download the list of all the products recommeded to various customers based on the proivious purchase decesions.

#lets read the file for all the recommedned products 
recom = pd.read_csv('C:\\Users\\Hp\\output\\Recommendation.csv')
recom.head()

recom.nunique()

We had 18,484 customers with 1,120 products recommended to these various customers based on previouse behaviours towards the products. If 70% (784 products) of this products recommended are venturally ordered AHG will grow sales by over 100% thereby increasing revenue and profitability by an appreciated level.




