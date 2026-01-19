import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Cusco region and elevation above 4700 meters
cusco_high_elevation = df[(df['region'] == 'cusco') & (df['elevation (m)'].astype(int) > 4700)]
# Calculate average elevation
avg_elevation = cusco_high_elevation['elevation (m)'].astype(int).mean()
print(f"Final Answer: {avg_elevation:.1f}")