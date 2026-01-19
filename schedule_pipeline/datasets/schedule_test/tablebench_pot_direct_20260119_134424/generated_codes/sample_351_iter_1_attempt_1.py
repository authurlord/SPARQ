import pandas as pd

df = pd.read_csv('table.csv')
# Remove the 'Total' and 'Source' rows
parties_df = df[df['Party'] != 'Total']
parties_df = parties_df[parties_df['Party'] != 'Source: http://www.stat.go.jp/data/chouki/27.htm']

# Convert '%' column to float
parties_df['%'] = pd.to_numeric(parties_df['%'])

# Count parties with more than 10% votes
count = (parties_df['%'] > 10).sum()
print(f"Final Answer: {count}")