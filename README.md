# Customer-Segmentation-using-K-Means-Algorithm
This project is about customer segmentation using K-Means Algorithm. Segmentation was done based on customer transactions data such as total payments, order frequency and order recency. Customers are segmented as loyal customers, active customers and at risk customers.

```python
# Import Packages
s = "Python syntax highlighting"
print s
```


### Import Packages
'''python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn import preprocessing
'''

### Data Preprocessing
'''
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
'''
'''
# Check total missing values in each datasets
df_names = ['df_cust', 'df_geo', 'df_order_items', 'df_order_payments', 'df_order_reviews', 'df_orders', 'df_prod_names', 'df_prod', 'df_sellers']
df_list = [df_cust, df_geo, df_order_items, df_order_payments, df_order_reviews, df_orders, df_prod_names, df_prod, df_sellers]

num_na = []
for i in range(9):
    na = df_list[i].isnull().sum().sum()
    num_na.append(na)

pd.DataFrame(list(zip(df_names, num_na)), columns = ['Dataset', 'Total Missing Value'])
'''
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
