import pandas as pd

# Load the data and skip the last row which contains source information
df = pd.read_csv('table.csv')
df = df.iloc[:-1]  # Remove the last row (source info)

# Convert the '%' column to float
df['%'] = pd.to_numeric(df['%'])

# Count parties with more than 10% of votes
count_parties = (df['%'] > 10).sum()

print(f"Final Answer: {count_parties}")