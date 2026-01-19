import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert columns to numeric, handling potential non-numeric entries
df['Research funding (£,000)'] = pd.to_numeric(df['Research funding (£,000)'], errors='coerce')
df['Total number of students'] = pd.to_numeric(df['Total number of students'].str.replace(',', ''), errors='coerce')

# Calculate average funding per student
df['Funding per Student (£)'] = df['Research funding (£,000)'] / df['Total number of students']

# Plot bar chart
plt.figure(figsize=(12, 6))
plt.bar(df['Institution'], df['Funding per Student (£)'], color='skyblue')
plt.title('Average Funding per Student by Institution')
plt.xlabel('Institution')
plt.ylabel('Funding per Student (£)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()