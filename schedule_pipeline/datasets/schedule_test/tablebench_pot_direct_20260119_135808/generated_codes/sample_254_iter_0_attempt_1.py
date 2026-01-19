import pandas as pd

df = pd.read_csv('table.csv')
# Filter districts in Cusco region with elevation above 4700 meters
filtered_df = df[(df['region'] == 'cusco') & (df['elevation (m)'].astype(int) > 4700)]
# Calculate average elevation
avg_elevation = filtered_df['elevation (m)'].astype(int).mean()
print(f"Final Answer: {avg_elevation:.1f}")