import pandas as pd               # for data manipulation
import matplotlib.pyplot as plt   # for plotting 
import seaborn as sns             # an extension of matplotlib for statistical graphics
## step 0 - import products.csv from directory '../input/products.csv'
products = pd.read_csv ('../input/products.csv')
## step 1 - .groupby( ) all available products (from products data frame) by their "aisle_id", then select to find the size of each group
aisle_top = products.groupby('aisle_id')[['aisle_id']].count()
aisle_top.head(10)
### step 2 - Rename the column of aisle_top as: 'total_products'
aisle_top.columns = ['total_products']
# Before you move on to step 3, have a look at your produced results so far.
# Check the results below
aisle_top.head()
## step 3 - Sort the values of total_products so to get the aisles with most products first.
aisle_top = aisle_top.sort_values(by='total_products', ascending=True)

## step 4 - Select the first 10 rows of the data frame. Remember that index in Python starts from 0
aisle_top = aisle_top.iloc[0 :9 ]

### Before you move on to the final step, how can you ensure that the aisle_top has only 10 aisles?
aisle_top.shape
# Have a look at the produced data frame before you plot it (visualize it).
# Are your results fine?
aisle_top.head(10)
## step 5 - Visualize the results. Place index on x-axis
plt.figure(figsize=(15,10))
sns.barplot(aisle_top.index, aisle_top.total_products, order=aisle_top.index)
plt.xlabel('Aisle_id', size=15)
plt.ylabel('Total_Products', size=15)
# Modify the limits
plt.ylim(0,100)
plt.show()
## step 0 - Import the order_products__prior.csv from directory '../input/order_products__prior.csv'
order_products_prior = pd.read_csv ('../input/order_products__prior.csv')
order_products_prior.head()
order_products_prior.shape

order_products_prior.head(10)
## step 1 - Filter order_products_prior and keep only these products with more than 30 purchases
avg_pos = order_products_prior.groupby('product_id').filter(lambda x: x.shape[0] >30)
avg_pos.head()
## step 2 -  .groupby( ) products and for add_to_cart_order column aggregate the values with the mean function.
avg_pos = avg_pos.groupby('product_id')[['add_to_cart_order']].mean()
avg_pos.head()
### step 3 - Rename column of avg_pos as: 'mean_add_to_cart_order'
avg_pos.columns = ['mean_add_to_cart_order']
## step 4 -  Use the proper method to sort the products by their mean_add_to_cart_order. Sort them in ascending order
avg_pos_asc = avg_pos.sort_values(by='mean_add_to_cart_order', ascending=True)
avg_pos_asc.head(10)
## step 5 - And now use again the same method to sort the products in descending order (store the results in a new DataFrame)
avg_pos_des = avg_pos.sort_values(by='mean_add_to_cart_order', ascending=False)
avg_pos_des.head()

## step 6 - Store the product_id of the product with the highest mean_add_to_cart_order
id_low = avg_pos_des.index[0]
## step 7 -  Import products.csv and find the name of the product with the highest mean_add_to_cart_order
products = pd.read_csv('../input/products.csv')
products[products.product_id== id_low ]
### step 8 - Create a sns.barplot for the 10 products with the lowest mean_add_to_cart_order
plt.figure(figsize=(12,8))
graph = sns.barplot(avg_pos_asc.index[0 : 9], avg_pos_asc['mean_add_to_cart_order'].iloc[0 : 9], order= avg_pos_asc.index[0 : 9])
plt.show()