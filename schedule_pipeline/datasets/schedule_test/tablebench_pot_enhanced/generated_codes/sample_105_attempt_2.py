import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert columns to numeric
df['Research funding (£,000)'] = pd.to_numeric(df['Research funding (£,000)'])
df['Total number of students'] = pd.to_numeric(df['Total number of students'])

# Calculate average funding per student
df['Funding per student (£)'] = df['Research funding (£,000)'] / df['Total number of students']

# Create bar chart
plt.figure(figsize=(12, 6))
plt.bar(df['Institution'], df['Funding per student (£)'], color='skyblue')
plt.title('Average Funding per Student for Each Institution')
plt.xlabel('Institution')
plt.ylabel('Funding per Student (£)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()