import pandas as pd

df = pd.read_csv('table.csv')
# Remove the total row (last row) as it is not a year-specific record
df_filtered = df[df['year'] != 'total']

# Calculate variance of 'wins' column
variance_wins = df_filtered['wins'].var()
print(f"Final Answer: {variance_wins:.2f}")