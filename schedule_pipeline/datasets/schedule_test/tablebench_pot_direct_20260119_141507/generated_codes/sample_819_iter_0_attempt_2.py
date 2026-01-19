import pandas as pd

df = pd.read_csv('table.csv')
# Filter couples who have performed more than 7 dances
filtered_df = df[df['number of dances'] > 7]
# Calculate variance of 'average' points
variance = filtered_df['average'].var()
print(f"Final Answer: {variance:.2f}")