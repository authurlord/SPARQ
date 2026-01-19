import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'passes through - district (s' contains 'sirmour'
filtered_df = df[df['passes through - district (s')'].str.contains('sirmour', case=False, na=False)]
# Extract the 'length (in km)' values
lengths = filtered_df['length (in km)'].astype(float)
# Calculate the difference between max and min length
length_difference = lengths.max() - lengths.min()
print(f"Final Answer: {length_difference:.2f}")