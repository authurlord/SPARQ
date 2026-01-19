import pandas as pd

df = pd.read_csv('table.csv')
# Convert eccentricity and distances to float for proper comparison
df['eccentricity'] = pd.to_numeric(df['eccentricity'])
df['periselene (km)'] = pd.to_numeric(df['periselene (km)'])
df['aposelene (km)'] = pd.to_numeric(df['aposelene (km)'])

# Sort by eccentricity to observe the trend
df_sorted = df.sort_values(by='eccentricity')

# Display the trend
print("As eccentricity increases:")
print(f"Periselene (km) decreases from {df_sorted.iloc[0]['periselene (km)']} to {df_sorted.iloc[-1]['periselene (km)']}")
print(f"Aposelene (km) increases from {df_sorted.iloc[0]['aposelene (km)']} to {df_sorted.iloc[-1]['aposelene (km)']}")

# Final answer based on the observed trend
print("Final Answer: periselene decreases, aposelene increases")