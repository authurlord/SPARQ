import pandas as pd

df = pd.read_csv('table.csv')
# Filter for mountains in Tanzania with elevation > 3000 and prominence < 3000
filtered_mountains = df[(df['country'] == 'tanzania') & 
                        (df['elevation (m)'].astype(int) > 3000) & 
                        (df['prominence (m)'].astype(int) < 3000)]
count = len(filtered_mountains)
print(f"Final Answer: {count}")