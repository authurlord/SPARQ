import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean land area column: extract numeric value after 'sq mi'
df['Land area'] = df['Land area'].str.extract(r'(\d+\.?\d*)').astype(float)

# Convert population to integer
df['Population (2012 est.)'] = df['Population (2012 est.)'].str.replace(',', '').astype(int)

# Calculate population density
df['density'] = df['Population (2012 est.)'] / df['Land area']

# Sort by density in descending order and take top 10
top_10 = df.sort_values(by='density', ascending=False).head(10)

# Create bar chart
plt.figure(figsize=(12, 6))
plt.bar(top_10['Name'], top_10['density'], color='skyblue')
plt.title('Top 10 Regions by Population Density')
plt.xlabel('Region')
plt.ylabel('Population Density (per sq mi)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the top 10 densities for reference
print(f"Final Answer: {top_10['density'].tolist()}")