import pandas as pd

# Load the data and exclude the last row which contains source information
df = pd.read_csv('table.csv')
df = df.iloc[:-1]  # Remove the last row (source)

# Convert the '%' column to float
df['%'] = pd.to_numeric(df['%'])

# Count parties with more than 10% votes
count = (df['%'] > 10).sum()

print(f"Final Answer: {count}")