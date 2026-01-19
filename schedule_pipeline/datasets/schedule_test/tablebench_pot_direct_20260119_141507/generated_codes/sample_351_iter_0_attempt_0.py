import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the "Total" row and keep only parties with >10% votes
filtered_df = df[df['%'].str.strip().astype(float) > 10.0]
# Exclude the "Total" row (which is at the end and has '%'=100)
# The row with 'Total' is the last one, so we drop it explicitly
filtered_df = filtered_df.drop(index=filtered_df.index[-1]) if len(filtered_df) > 0 else filtered_df
# Count the number of parties with more than 10%
party_count = len(filtered_df)
print(f"Final Answer: {party_count}")