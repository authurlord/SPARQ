import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'obama%' to numeric, removing the '%' sign
df['obama%'] = df['obama%'].str.replace('%', '').astype(float)
# Filter counties where Obama received more than 60%
filtered_counties = df[df['obama%'] > 60]
# Sum the 'total' column for these counties
total_votes = filtered_counties['total'].sum()
print(f"Final Answer: {total_votes}")