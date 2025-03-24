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
