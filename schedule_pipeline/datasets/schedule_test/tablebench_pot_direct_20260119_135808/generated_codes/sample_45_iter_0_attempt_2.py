import pandas as pd

df = pd.read_csv('table.csv')
# Convert Start and End to integers
df['Start'] = pd.to_numeric(df['Start'])
df['End'] = pd.to_numeric(df['End'])

# Calculate the number of years served for each ambassador
df['Years'] = df['End'] - df['Start']

# Sum the total years served by all ambassadors
total_years = df['Years'].sum()
print(f"Final Answer: {total_years}")