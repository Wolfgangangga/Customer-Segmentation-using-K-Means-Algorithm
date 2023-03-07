# Customer-Segmentation-using-K-Means-Algorithm
This project is about customer segmentation using K-Means Algorithm. Segmentation was done based on customer transactions data such as total payments, order frequency and order recency. Customers are segmented as loyal customers, active customers and at risk customers.

### Import Packages
```python
# Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn import preprocessing
```

### Data Preprocessing
```python
# Import Dataset
# Customers dataset
url = 'https://drive.google.com/file/d/18Xz9KswZaMuE0sr90PDU62huvM9n6Jdd/view?usp=sharing'
file_id = url.split("/")[-2]
read_url = "https://drive.google.com/uc?id=" + file_id
df_cust = pd.read_csv(read_url)

# Geolocation dataset
url = 'https://drive.google.com/file/d/1y1WIXnHPsBMXrLThmfI9loxx9Wmj4u3H/view?usp=sharing'
file_id = url.split("/")[-2]
read_url = "https://drive.google.com/uc?id=" + file_id
df_geo = pd.read_csv(read_url)

# Order items dataset 
url = 'https://drive.google.com/file/d/1iwp1UqYUi6fQU4UVnw-hrhil2aGRKY5n/view?usp=sharing'
file_id = url.split("/")[-2]
read_url = "https://drive.google.com/uc?id=" + file_id
df_order_items = pd.read_csv(read_url)

# Order payments dataset 
url = 'https://drive.google.com/file/d/1A0GvK_TpfMq4EcfL5I9EkyxAcFxW-Pab/view?usp=sharing'
file_id = url.split("/")[-2]
read_url = "https://drive.google.com/uc?id=" + file_id
df_order_payments = pd.read_csv(read_url)

# Order reviews dataset 
url = 'https://drive.google.com/file/d/1InDHJFl5S7z_p5IwYcSO8K7h69Ll-hRI/view?usp=sharing'
file_id = url.split("/")[-2]
read_url = "https://drive.google.com/uc?id=" + file_id
df_order_reviews = pd.read_csv(read_url)

# Orders dataset 
url = 'https://drive.google.com/file/d/1YVVpwk-3yz31gAAUF1h_2FMOnRrTSLZi/view?usp=sharing'
file_id = url.split("/")[-2]
read_url = "https://drive.google.com/uc?id=" + file_id
df_orders = pd.read_csv(read_url)

# Product category name translation dataset 
url = 'https://drive.google.com/file/d/1Jx5-Y4lEQtnwCUzdmTUOPdy0AO9RJe7H/view?usp=sharing'
file_id = url.split("/")[-2]
read_url = "https://drive.google.com/uc?id=" + file_id
df_prod_names = pd.read_csv(read_url)

# Products dataset 
url = 'https://drive.google.com/file/d/106ETFTrUh79e-PSNiG8W_NtJ5kvre9np/view?usp=sharing'
file_id = url.split("/")[-2]
read_url = "https://drive.google.com/uc?id=" + file_id
df_prod = pd.read_csv(read_url)

# Sellers dataset 
url = 'https://drive.google.com/file/d/1SvjWZwLkMYpmbnqMFADAosaosjyxZP52/view?usp=sharing'
file_id = url.split("/")[-2]
read_url = "https://drive.google.com/uc?id=" + file_id
df_sellers = pd.read_csv(read_url)
```
```python
# Check total missing values in each datasets
df_names = ['df_cust', 'df_geo', 'df_order_items', 'df_order_payments', 'df_order_reviews', 'df_orders', 'df_prod_names', 'df_prod', 'df_sellers']
df_list = [df_cust, df_geo, df_order_items, df_order_payments, df_order_reviews, df_orders, df_prod_names, df_prod, df_sellers]

num_na = []
for i in range(9):
    na = df_list[i].isnull().sum().sum()
    num_na.append(na)

pd.DataFrame(list(zip(df_names, num_na)), columns = ['Dataset', 'Total Missing Value'])
```
| Dataset           | Total Missing Value   |
| ------------------|----------------------:|
| df_cust           |                      0|
| df_geo            |                      0|
| df_order_items    |                      0|
| df_order_payments |                      0|
| df_order_reviews  |                 145903|
| df_orders         |                   4908|
| df_prod_names     |                      0|
| df_prod           |                   2448|
| df_sellers        |                      0|

Based on the table above, could be known that there are missing values on 'df_order_reviews', 'df_orders', and 'df_prod' datasets. To avoid the missing values effects on the analysis, each rows that contain missing value(s) will be removed from the datasets.

```python
# Drop rows that contains missing value(s) in each datasets
for i in range(9):
    df_list[i].dropna(inplace = True)
    
# Check datasets still contain missing values or not
num_na = []
for i in range(9):
    na = df_list[i].isnull().sum().sum()
    num_na.append(na)

pd.DataFrame(list(zip(df_names, num_na)), columns = ['Dataset', 'Total Missing Value'])
```
| Dataset           | Total Missing Value   |
| ------------------|----------------------:|
| df_cust           |                      0|
| df_geo            |                      0|
| df_order_items    |                      0|
| df_order_payments |                      0|
| df_order_reviews  |                      0|
| df_orders         |                      0|
| df_prod_names     |                      0|
| df_prod           |                      0|
| df_sellers        |                      0|

Based on the table above, we know that there aren't missing values anymore in each datasets.

```python
# Combine datasets into a dataframe
main_df = pd.merge(df_cust, df_orders)
main_df = pd.merge(main_df, df_order_items)
main_df = pd.merge(main_df, df_prod)
main_df = pd.merge(main_df, df_prod_names)
main_df = pd.merge(main_df, df_order_payments)
main_df = pd.merge(main_df, df_order_reviews)
main_df = pd.merge(main_df, df_sellers)
```

```python
# Check Overview of merged dataset
main_df.info()
```
```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 11578 entries, 0 to 11577
Data columns (total 40 columns):
 #   Column                         Non-Null Count  Dtype  
---  ------                         --------------  -----  
 0   customer_id                    11578 non-null  object 
 1   customer_unique_id             11578 non-null  object 
 2   customer_zip_code_prefix       11578 non-null  int64  
 3   customer_city                  11578 non-null  object 
 4   customer_state                 11578 non-null  object 
 5   order_id                       11578 non-null  object 
 6   order_status                   11578 non-null  object 
 7   order_purchase_timestamp       11578 non-null  object 
 8   order_approved_at              11578 non-null  object 
 9   order_delivered_carrier_date   11578 non-null  object 
 10  order_delivered_customer_date  11578 non-null  object 
 11  order_estimated_delivery_date  11578 non-null  object 
 12  order_item_id                  11578 non-null  int64  
 13  product_id                     11578 non-null  object 
 14  seller_id                      11578 non-null  object 
 15  shipping_limit_date            11578 non-null  object 
 16  price                          11578 non-null  float64
 17  freight_value                  11578 non-null  float64
 18  product_category_name          11578 non-null  object 
 19  product_name_lenght            11578 non-null  float64
 20  product_description_lenght     11578 non-null  float64
 21  product_photos_qty             11578 non-null  float64
 22  product_weight_g               11578 non-null  float64
 23  product_length_cm              11578 non-null  float64
 24  product_height_cm              11578 non-null  float64
 25  product_width_cm               11578 non-null  float64
 26  product_category_name_english  11578 non-null  object 
 27  payment_sequential             11578 non-null  int64  
 28  payment_type                   11578 non-null  object 
 29  payment_installments           11578 non-null  int64  
 30  payment_value                  11578 non-null  float64
 31  review_id                      11578 non-null  object 
 32  review_score                   11578 non-null  int64  
 33  review_comment_title           11578 non-null  object 
 34  review_comment_message         11578 non-null  object 
 35  review_creation_date           11578 non-null  object 
 36  review_answer_timestamp        11578 non-null  object 
 37  seller_zip_code_prefix         11578 non-null  int64  
 38  seller_city                    11578 non-null  object 
 39  seller_state                   11578 non-null  object 
dtypes: float64(10), int64(6), object(24) 
```

Based on the information above, the merged dataset has dimension 11577 x 40. There aren't missing values on the merged dataset. However, the datetime-based columns are not in the correct data type.

```python
# Convert several columns into datetime type
time_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date',
             'order_estimated_delivery_date', 'shipping_limit_date', 'review_creation_date', 'review_answer_timestamp']
main_df[time_cols] = main_df[time_cols].apply(pd.to_datetime)
main_df.info()
```

```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 11578 entries, 0 to 11577
Data columns (total 40 columns):
 #   Column                         Non-Null Count  Dtype         
---  ------                         --------------  -----         
 0   customer_id                    11578 non-null  object        
 1   customer_unique_id             11578 non-null  object        
 2   customer_zip_code_prefix       11578 non-null  int64         
 3   customer_city                  11578 non-null  object        
 4   customer_state                 11578 non-null  object        
 5   order_id                       11578 non-null  object        
 6   order_status                   11578 non-null  object        
 7   order_purchase_timestamp       11578 non-null  datetime64[ns]
 8   order_approved_at              11578 non-null  datetime64[ns]
 9   order_delivered_carrier_date   11578 non-null  datetime64[ns]
 10  order_delivered_customer_date  11578 non-null  datetime64[ns]
 11  order_estimated_delivery_date  11578 non-null  datetime64[ns]
 12  order_item_id                  11578 non-null  int64         
 13  product_id                     11578 non-null  object        
 14  seller_id                      11578 non-null  object        
 15  shipping_limit_date            11578 non-null  datetime64[ns]
 16  price                          11578 non-null  float64       
 17  freight_value                  11578 non-null  float64       
 18  product_category_name          11578 non-null  object        
 19  product_name_lenght            11578 non-null  float64       
 20  product_description_lenght     11578 non-null  float64       
 21  product_photos_qty             11578 non-null  float64       
 22  product_weight_g               11578 non-null  float64       
 23  product_length_cm              11578 non-null  float64       
 24  product_height_cm              11578 non-null  float64       
 25  product_width_cm               11578 non-null  float64       
 26  product_category_name_english  11578 non-null  object        
 27  payment_sequential             11578 non-null  int64         
 28  payment_type                   11578 non-null  object        
 29  payment_installments           11578 non-null  int64         
 30  payment_value                  11578 non-null  float64       
 31  review_id                      11578 non-null  object        
 32  review_score                   11578 non-null  int64         
 33  review_comment_title           11578 non-null  object        
 34  review_comment_message         11578 non-null  object        
 35  review_creation_date           11578 non-null  datetime64[ns]
 36  review_answer_timestamp        11578 non-null  datetime64[ns]
 37  seller_zip_code_prefix         11578 non-null  int64         
 38  seller_city                    11578 non-null  object        
 39  seller_state                   11578 non-null  object        
dtypes: datetime64[ns](8), float64(10), int64(6), object(16)
```

### Clustering Analysis: Customers Segmentation
#### Data Preprocessing for Clustering Analysis
```python
# Get the last purchase date were made by customers
last_purchase_date = main_df.order_purchase_timestamp.max()
last_purchase_date
```
```
Timestamp('2018-08-29 14:18:28')
```

```python
# Assume the present date is the first date of the next month of customers last purchase date
present_date = datetime(2018, 9, 1)
present_date
```
```
datetime.datetime(2018, 9, 1, 0, 0)
```

```python
# Prepare the dataset for customers segmentation
df_clustering = main_df.groupby('customer_unique_id').agg({'customer_state': lambda x: x.max(),
                                                           'payment_value': lambda x: x.sum(),
                                                           'order_id': lambda x: len(x),
                                                           'order_purchase_timestamp': lambda x: (present_date - x.max()).days})

df_clustering.columns = ['State', 'Total_Payment', 'Order_Frequency', 'Order_Recency']
print(df_clustering.info())
df_clustering
```
```
<class 'pandas.core.frame.DataFrame'>
Index: 9333 entries, 0000366f3b9a7992bf8c76cfdf3221e2 to ffff5962728ec6157033ef9805bacc48
Data columns (total 4 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   State            9333 non-null   object 
 1   Total_Payment    9333 non-null   float64
 2   Order_Frequency  9333 non-null   int64  
 3   Order_Recency    9333 non-null   int64  
dtypes: float64(1), int64(2), object(1)
memory usage: 364.6+ KB
None
```
| customer_unique_id               | State | Total_Payment | Order_Frequency | Order_Recency | 
| -------------------------------- |:-----:| -------------:| ---------------:| -------------:| 
| 0000366f3b9a7992bf8c76cfdf3221e2 | SP    | 141.90	       | 1	             | 113           |
| 000ec5bff359e1c0ad76a81a45cb598f | SP	   | 27.75	       | 1	             | 10            |
| 00172711b30d52eea8b313a7f2cced02 | BA	   | 122.07	       | 1	             | 34            |
| 001928b561575b2821c92254a2327d06 | SP	   | 329.62	       | 2	             | 7             |
| ...                              | ...   | ...	       | ...             | ...           |		
| fff3e1d7bc75f11dc7670619b2e61840 | PI	   | 82.51	       | 1	             | 42            |
| ffff5962728ec6157033ef9805bacc48 | ES	   | 133.69	       | 1	             | 121           |

For clustering analysis, a dataset is created using informations from the merged dataset. The dataset contains location, total payment, order frequency, and order recency for each customers. The number of customers are 9333. The dataset have no missing values.

#### EDA of Clustering Dataset
```python
# Visualisation for each columns
print(df_clustering.describe())

fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize=(20, 5))

sns.countplot(x = df_clustering['State'], order=pd.value_counts(df_clustering['State']).iloc[:10].index, ax = ax[0])
ax[1].boxplot(x = df_clustering['Total_Payment'], whis=1.5)
ax[2].boxplot(x = df_clustering['Order_Frequency'], whis=1.5)
ax[3].boxplot(x = df_clustering['Order_Recency'], whis=1.5)
ax[1].set(title = 'Total Payment', xticklabels=[])
ax[2].set(title = 'Order Frequency', xticklabels=[])
ax[3].set(title = 'Order Recency', xticklabels=[])

plt.show()
```
```
       Total_Payment  Order_Frequency  Order_Recency
count    9333.000000      9333.000000    9333.000000
mean      245.638111         1.240544      75.423551
std       641.607276         0.723361      42.535458
min        13.890000         1.000000       2.000000
25%        68.180000         1.000000      37.000000
50%       122.420000         1.000000      75.000000
75%       217.550000         1.000000     113.000000
max     29099.520000        13.000000     484.000000
```

![alt text](https://github.com/Wolfgangangga/Customer-Segmentation-using-K-Means-Algorithm/blob/main/EDA%201.png)

From the visualisation analysis above, could be known that most of the customers live in SP, RJ and MG state. About 75% of the customers have total payment below 220 and order frequency of 1, they can be considered as new customer. About 25% of the customers have recently made a transaction in the last 37 days.

```python
# Visualisation between customer location and others variables
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

sns.boxplot(data = df_clustering, x = 'State', y = 'Total_Payment', order = pd.value_counts(df_clustering['State']).iloc[:10].index, ax = ax[0], whis=1.5)
sns.boxplot(data = df_clustering, x = 'State', y = 'Order_Frequency', order = pd.value_counts(df_clustering['State']).iloc[:10].index, ax = ax[1], whis=1.5)
sns.boxplot(data = df_clustering, x = 'State', y = 'Order_Recency', order = pd.value_counts(df_clustering['State']).iloc[:10].index, ax = ax[2], whis=1.5)

plt.show()
```

![alt text](https://github.com/Wolfgangangga/Customer-Segmentation-using-K-Means-Algorithm/blob/main/EDA%202.png)

In this visualisation, the visualisations only use top 10 state that have the most customers assuming its can represent other states. From the plots above, could be known that total payment, order frequency and order recency have the same pattern in all of the states. Therefore, 'State' variable is not informative to be used on customer segmentation.

#### Clustering using K-Means Algorithm
Before do clustering using K-Means algorithm, each columns in the dataset should be scaled to create same scale data. The scaling process is needed because K-Means algorithm use distance-based dissimilarity measure. Besides, the scaling process can solve outliers problem.

```python
# Scaling dataset
df = df_clustering.drop(['State'], axis = 1)
scaler = preprocessing.MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns = ['Total_Payment', 'Order_Frequency', 'Order_Recency'])
df
```

| Total_Payment | Order_Frequency |	Order_Recency |
| -------------:| ---------------:| -------------:|
| 0.004401   	| 0.000000   	  | 0.230290      |
| 0.000477   	| 0.000000  	  | 0.016598      |
| 0.003719  	| 0.000000  	  | 0.066390      |
| 0.010855  	| 0.083333  	  | 0.010373      |
| ...	        | ...             |	...	          |
| 0.002359  	| 0.000000  	  | 0.082988      |
| 0.004119  	| 0.000000  	  | 0.246888      |

```python
# Find the best number of clusters
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 5))

# Elbow Method
from sklearn.cluster import KMeans
sse = []
for k in range(1, 11):
    kmeans = KMeans(init = 'random', n_clusters = k, n_init = 10, max_iter = 300, random_state = 123)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
    
plt.style.use("fivethirtyeight")
ax[0].plot(range(1, 11), sse)
ax[0].xaxis.set(ticks = range(1, 11))
ax[0].set_xlabel("Number of Clusters")
ax[0].set_ylabel("SSE")

# Silhouette Score
from sklearn.metrics import silhouette_score
silhouette_coefficients = []
for k in range(2, 11):
    kmeans = KMeans(init = 'random', n_clusters = k, n_init = 10, max_iter = 300, random_state = 123)
    kmeans.fit(df)
    score = silhouette_score(df, kmeans.labels_)
    silhouette_coefficients.append(score)
    
plt.style.use("fivethirtyeight")
ax[1].plot(range(2, 11), silhouette_coefficients)
ax[1].set_xlabel("Number of Clusters")
ax[1].set_ylabel("Silhouette Coefficient")

plt.show()
```

![alt text](https://github.com/Wolfgangangga/Customer-Segmentation-using-K-Means-Algorithm/blob/main/Elbow%20and%20Silhouette.png)

From the plots above, could be known that the best number of cluster is k = 3 because it is the smallest number of clusters that has small error and highest sillhoutte score.

#### Customers Segemntation Model
```python
# Model
cust_seg = KMeans(init = 'random', n_clusters = 3, n_init = 10, max_iter = 300, random_state = 123)
cust_seg_model = cust_seg.fit(df)
```

```python
# Cluster Centers
cluster = ['Cluster 1', 'Cluster 2', 'Cluster 3']
pd.DataFrame(cust_seg_model.cluster_centers_, columns = ['Total Payment', 'Order Frequency', 'Order Recency'], index = cluster)
```

|           | Total Payment | Order Frequency |	Order Recency |
| --------- | -------------:| ---------------:| -------------:|
| Cluster 1	| 0.051255  	| 0.257042  	  | 0.167150      |
| Cluster 2	| 0.006419	    | 0.010679	      | 0.228657      |
| Cluster 3	| 0.006097	    | 0.010669	      | 0.076494      |

From the table above, customers can be segmented as follows:
Cluster 1 as Loyal Customers
Customers from this cluster have large transaction payments and make pruchases more often.
Cluster 2 as At Risk Customers
Customers from this cluster have small transaction payments and have not purchases for a long time.
Cluster 3 as Active Customers
Customers from this cluster recently make purchases.

#### Segmentation New Customer
Scaling customer's data using Min-Max Scaling method:

Total Payment
Total Payment_scaled = (Total Payment - 13.89)/(29099.52 - 13.89)

Order Frequency
Order Frequency_scaled = (Order Frequency - 1)/(13 - 1)

Order Recency
Order Recency_scaled = (Order Recency - 2)/(484 - 2)


Calculate distance measure for clustering
distance = sqrt((Total Payment_scaled - Total Payment_centroid cluster i)^2 + (Order Frequency_scaled - Order Frequency_centroid cluster i)^2 + (Order Recency_scaled - Order Recency_centroid cluster i)^2)

for i : cluster 1, cluster 2, cluster 3


The above formula is applied to each clusters.

Then get distance of new customer to each clusters and find the smallest ones. The smallest distance is the cluster for the new customer.
