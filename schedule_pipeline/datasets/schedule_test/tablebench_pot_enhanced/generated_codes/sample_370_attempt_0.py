import pandas as pd

df = pd.read_csv('table.csv')
# Convert elevation and prominence to numeric for comparison
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'])
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'])

# Filter for Tanzania, elevation > 3000, and prominence < 3000
tanzania_mountains = df[df['country'] == 'tanzania']
filtered_mountains = tanzania_mountains[
    (tanzania_mountains['elevation (m)'] > 3000) &
    (tanzania_mountains['prominence (m)'] < 3000)
]

# Count the number of such mountains
count = len(filtered_mountains)
print(f"Final Answer: {count}")