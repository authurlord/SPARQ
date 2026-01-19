import pandas as pd

df = pd.read_csv('table.csv')
# Convert elevation and prominence to integers
df['elevation (m)'] = df['elevation (m)'].str.replace(',', '').astype(int)
df['prominence (m)'] = df['prominence (m)'].str.replace(',', '').astype(int)

# Filter mountains in Tanzania with elevation > 3000 and prominence < 3000
tanzania_mountains = df[(df['country'] == 'tanzania') & 
                        (df['elevation (m)'] > 3000) & 
                        (df['prominence (m)'] < 3000)]

count = len(tanzania_mountains)
print(f"Final Answer: {count}")