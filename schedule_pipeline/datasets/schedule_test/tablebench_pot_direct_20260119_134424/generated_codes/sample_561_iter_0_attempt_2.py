import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 1942 to 1946
df_filtered = df[df['Year'].astype(int).between(1942, 1946)]
# Extract US Chart position and clean non-numeric values
df_filtered['US Chart position'] = df_filtered['US Chart position'].str.replace(r'\(.*\)', '', regex=True).astype(float)
# Calculate average
avg_position = df_filtered['US Chart position'].mean()
print(f"Final Answer: {avg_position:.1f}")