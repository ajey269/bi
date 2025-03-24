Aim: Write a Python program to read data from a CSV file, perform
simple data analysis, and generate basic insights. (Use Pandas is a
Python library).


import pandas as pd
file_path = 'data.csv'
data = pd.read_csv(file_path)
print("First 5 rows ofthe dataset:")
print(data.head())
print("\nDataset Information:")
print(data.info())
print("\nSummary Statistical:")
print(data.describe())
if 'Category' in data.columns:
print("\nUnique valuesin 'Category' column:")
print(data['Category'].value_counts())



Practical 7:
 Write a Python program to read data from a CSV file, perform simple data analysis, and generate basic insights. (Use Pandas is a Python library)

import pandas as pd

df = pd.read_csv('student.csv') 
print("Columns:", df.columns)

df.columns = df.columns.str.strip()
print(df.info())
print(df.head())

print(df.select_dtypes(include='number').describe())
print(df.isnull().sum())

print(df.select_dtypes(include='number').corr())

import matplotlib.pyplot as plt
df['Score'].hist(bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Score')
plt.show()

