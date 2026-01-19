import pandas as pd

df = pd.read_csv('table.csv')
# Filter districts in Cusco with elevation above 4700 meters
filtered_df = df[(df['region'] == 'cusco') & (df['elevation (m)'] > 4700)]
# Calculate average elevation
average_elevation = filtered_df['elevation (m)'].mean()
print(f"Final Answer: {average_elevation:.1f}")