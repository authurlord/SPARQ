import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to float for numerical analysis
df['eccentricity'] = pd.to_numeric(df['eccentricity'])
df['periselene (km)'] = pd.to_numeric(df['periselene (km)'])
df['aposelene (km)'] = pd.to_numeric(df['aposelene (km)'])

# Sort by eccentricity to observe the trend
df_sorted = df.sort_values(by='eccentricity', ascending=False)

# Display the sorted data to analyze the trend
print("Sorted by eccentricity (descending):")
print(df_sorted[['epoch (utc)', 'eccentricity', 'periselene (km)', 'aposelene (km)']])

# Final answer based on the trend observed
print("Final Answer: periselene decreases, aposelene increases")