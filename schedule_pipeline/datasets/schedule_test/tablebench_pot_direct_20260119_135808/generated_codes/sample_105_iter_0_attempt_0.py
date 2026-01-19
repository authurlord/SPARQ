import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Clean and convert columns
df['Research funding (£,000)'] = df['Research funding (£,000)'].str.replace(',', '').astype(float)
df['Total number of students'] = df['Total number of students'].str.replace(',', '').astype(float)

# Calculate average funding per student
df['Funding per Student (£)'] = df['Research funding (£,000)'] * 1000 / df['Total number of students']

# Create bar chart
plt.figure(figsize=(12, 6))
plt.bar(df['Institution'], df['Funding per Student (£)'], color='skyblue')
plt.title('Average Funding per Student for Universities')
plt.xlabel('Institution')
plt.ylabel('Funding per Student (£)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()