import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'number of dances' to integer
df['number of dances'] = pd.to_numeric(df['number of dances'], errors='coerce')

# Filter couples who have performed more than 7 dances
filtered_avg = df[df['number of dances'] > 7]['average']

# Compute variance of the 'average' points
variance = filtered_avg.var()

print(f"Final Answer: {variance:.2f}")