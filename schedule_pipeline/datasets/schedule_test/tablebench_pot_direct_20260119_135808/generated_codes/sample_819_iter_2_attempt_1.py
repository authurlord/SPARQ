import pandas as pd

df = pd.read_csv('table.csv')
# Filter couples who performed more than 7 dances
filtered_df = df[df['number of dances'] > 7]
# Calculate variance of the 'average' column
variance = filtered_df['average'].var()
print(f"Final Answer: {variance:.2f}")