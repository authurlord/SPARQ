import pandas as pd

df = pd.read_csv('table.csv')
# Filter for mountains in Papua New Guinea with elevation >= 3000 meters
filtered_mountains = df[(df['country'] == 'papua new guinea') & (df['elevation (m)'].astype(int) >= 3000)]
count = len(filtered_mountains)
print(f"Final Answer: {count}")