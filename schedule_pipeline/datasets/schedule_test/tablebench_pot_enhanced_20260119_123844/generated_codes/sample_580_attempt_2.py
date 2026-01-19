import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'bötzow' column to float
df['bötzow'] = df['bötzow'].astype(float)
# Calculate the difference between consecutive years
df['increase'] = df['bötzow'].diff()
# Find the year with the maximum increase
max_increase_year = df.loc[df['increase'].idxmax(), 'year']
print(f"Final Answer: {max_increase_year}")