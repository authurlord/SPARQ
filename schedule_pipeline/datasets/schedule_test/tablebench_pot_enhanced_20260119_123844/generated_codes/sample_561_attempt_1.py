import pandas as pd

df = pd.read_csv('table.csv')
# Filter data from 1942 to 1946
df_filtered = df[(df['Year'] >= '1942') & (df['Year'] <= '1946')]
# Extract and clean US Chart position
df_filtered['US Chart position'] = pd.to_numeric(df_filtered['US Chart position'], errors='coerce')
# Calculate average
avg_chart_position = df_filtered['US Chart position'].mean()
print(f"Final Answer: {avg_chart_position:.1f}")