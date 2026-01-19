import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the data
df = pd.read_csv('table.csv')

# Clean land area column to extract numeric value
def extract_land_area(area_str):
    # Extract number before 'sq mi' using regex
    match = re.search(r'(\d+\.?\d*)\s+sq\s+mi', area_str)
    return float(match.group(1)) if match else 0

# Apply cleaning to land area column
df['land_area'] = df['Land area'].apply(extract_land_area)

# Convert population to numeric (remove commas)
df['population_2012'] = pd.to_numeric(df['Population (2012 est.)'].str.replace(',', ''), errors='coerce')

# Calculate population density
df['density'] = df['population_2012'] / df['land_area']

# Remove rows where density is 0 or NaN
df = df[df['density'] > 0]

# Sort by density in descending order and take top 10
top_10 = df.nlargest(10, 'density')[['Name', 'density']]

# Create bar chart
plt.figure(figsize=(12, 8))
plt.bar(top_10['Name'], top_10['density'], color='skyblue')
plt.title('Top 10 Regions by Population Density')
plt.xlabel('Region')
plt.ylabel('Population Density (people per sq mi)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Print the final answer as requested
print(f"Final Answer: {top_10['Name'].tolist()}")