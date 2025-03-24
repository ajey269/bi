Practical 8.a)
 Perform data visualization using Python on any sales data.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05', 
             '2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09', '2025-01-10'],
    'Product': ['Product A', 'Product B', 'Product A', 'Product C', 'Product B', 
                'Product A', 'Product C', 'Product B', 'Product A', 'Product C'],
    'Sales': [50, 30, 70, 90, 60, 80, 50, 40, 60, 30],
    'Revenue': [500, 450, 700, 900, 900, 800, 750, 600, 600, 450]
}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.barplot(x='Product', y='Sales', data=df, palette='viridis')
plt.title('Total Sales per Product')

plt.subplot(2, 2, 2)
sns.lineplot(x='Date', y='Sales', data=df, marker='o', color='b')
plt.title('Sales Trend Over Time')
product_sales = df.groupby('Product')['Sales'].sum()
plt.subplot(2, 2, 3)
product_sales.plot.pie(autopct='%1.1f%%', startangle=90, cmap='Set3')
plt.title('Sales Distribution by Product')

plt.subplot(2, 2, 4)
sns.scatterplot(x='Sales', y='Revenue', data=df, hue='Product', palette='deep')
plt.title('Sales vs Revenue')

plt.tight_layout()
plt.show()


