import pandas as pd

df = pd.read_csv('table.csv')
# Convert Start and End to integers
df['Start'] = pd.to_numeric(df['Start'])
df['End'] = pd.to_numeric(df['End'])

# Calculate years served for each ambassador
df['Years Served'] = df['End'] - df['Start']

# Sum the total years served by all ambassadors
total_years = df['Years Served'].sum()
print(f"Final Answer: {total_years}")