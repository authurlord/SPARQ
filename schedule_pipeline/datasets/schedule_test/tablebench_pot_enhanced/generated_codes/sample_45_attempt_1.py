import pandas as pd

df = pd.read_csv('table.csv')
# Convert Start and End to integers
df['Start'] = df['Start'].astype(int)
df['End'] = df['End'].astype(int)
# Calculate years served for each ambassador
df['Years'] = df['End'] - df['Start'] + 1
# Sum total years served
total_years = df['Years'].sum()
print(f"Final Answer: {total_years}")