import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total s ton' to integer
df['total s ton'] = df['total s ton'].astype(int)
# Calculate the year-on-year increase
df['increase'] = df['total s ton'].diff()
# Find the row with the maximum increase
max_increase_row = df.loc[df['increase'].idxmax()]
# Extract the year with the highest increase
year_with_highest_increase = max_increase_row['year']
print(f"Final Answer: {year_with_highest_increase}")