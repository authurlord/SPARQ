import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the "Total" row and select only parties with >10% votes
filtered_df = df[df['%'] > '10.0'].dropna()
# Exclude the "Total" row by checking if the party name is not "Total"
party_filter = df['Party'] != 'Total'
filtered_df = df[party_filter & (df['%'] > '10.0')]
count = filtered_df.shape[0]
print(f"Final Answer: {count}")