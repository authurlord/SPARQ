import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the 'total' row and select only the years 2004 to 2010
tries_data = df[df['year'] != 'total']['tries'].astype(int)
# Calculate variance
variance_tries = tries_data.var()
print(f"Final Answer: {variance_tries:.2f}")