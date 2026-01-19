import pandas as pd

df = pd.read_csv('table.csv')
# Filter for mountains in Tanzania
tanzania_mountains = df[df['country'] == 'tanzania']
# Apply conditions: elevation > 3000 and prominence < 3000
filtered_mountains = tanzania_mountains[
    (tanzania_mountains['elevation (m)'].astype(int) > 3000) &
    (tanzania_mountains['prominence (m)'].astype(int) < 3000)
]
# Count the number of such mountains
count = len(filtered_mountains)
print(f"Final Answer: {count}")