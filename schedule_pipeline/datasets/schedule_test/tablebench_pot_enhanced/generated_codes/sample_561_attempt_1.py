import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 1942 to 1946
df_filtered = df[(df['Year'] >= '1942') & (df['Year'] <= '1946')]
# Extract numeric part from 'US Chart position'
df_filtered['US Chart position'] = df_filtered['US Chart position'].str.extract('(\d+)').astype(float)
# Calculate average US Chart position
avg_position = df_filtered['US Chart position'].mean()
print(f"Final Answer: {avg_position:.1f}")