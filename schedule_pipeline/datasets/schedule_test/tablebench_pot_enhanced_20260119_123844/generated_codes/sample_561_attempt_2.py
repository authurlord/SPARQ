import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 1942 to 1946
df_filtered = df[(df['Year'] >= '1942') & (df['Year'] <= '1946')]

# Clean 'US Chart position' by extracting only numeric values
df_filtered['US Chart position'] = df_filtered['US Chart position'].astype(str).str.extract('(\d+)').astype(float)

# Calculate average US Chart position
avg_position = df_filtered['US Chart position'].mean()

print(f"Final Answer: {avg_position:.1f}")