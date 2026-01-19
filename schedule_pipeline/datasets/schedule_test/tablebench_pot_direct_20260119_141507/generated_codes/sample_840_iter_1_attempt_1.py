import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'passes through - district (s' contains 'sirmour'
filtered_df = df[df['passes through - district (s)'].str.contains('sirmour', case=False, na=False)]
# Extract lengths and find max and min
lengths = filtered_df['length (in km)'].astype(float)
max_length = lengths.max()
min_length = lengths.min()
difference = max_length - min_length
print(f"Final Answer: {difference:.2f}")