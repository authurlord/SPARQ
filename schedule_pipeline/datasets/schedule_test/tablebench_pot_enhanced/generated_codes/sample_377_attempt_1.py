import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'elevation (m)' to integer for comparison
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
# Filter for mountains in Papua New Guinea with elevation >= 3000 meters
count = df[(df['country'] == 'papua new guinea') & (df['elevation (m)'] >= 3000)].shape[0]
print(f"Final Answer: {count}")