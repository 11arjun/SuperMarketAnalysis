import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder

# Loading DataSet
Data = read_csv('supermarket_sale.csv')
pd.set_option('display.max_columns', None)  # Shows all columns
pd.set_option('display.max_rows', None)     # Shows all rows
pd.set_option('display.max_colwidth', None) # Shows full column content
# Displaying basic information  about the Dataset
print(Data.info())
print(Data.describe())
# Checking missing values
missing_values = Data.isnull().sum()
# printing Missing values
print(missing_values)
# Converting Date and Time  columns
Data['Date'] = pd.to_datetime(Data['Date'])
Data['Time'] = pd.to_datetime(Data['Time'],format = '%H:%M').dt.time
# Identify nominal attributes (categorical columns)
nominal_columns = Data.select_dtypes(include=['object']).columns
print("Nominal Attributes" ,nominal_columns )
# Converting nominal data to numerical data , Using Label Encoding
label_encoders = {}
for column in nominal_columns:
    le =  LabelEncoder()
    Data[column] = le.fit_transform(Data[column])
    label_encoders[column] = le
print("ALL datas", Data.head())

