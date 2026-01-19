import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the "Total" row and select only parties with more than 10% votes
filtered_df = df[df['%'] > '10.0'].dropna(subset=['%'])
# Exclude the "Total" row by checking if the party name is not "Total"
filtered_df = filtered_df[filtered_df['Party'] != 'Total']
# Count the number of such parties
party_count = len(filtered_df)
print(f"Final Answer: {party_count}")