import pandas as pd

df = pd.read_csv('table.csv')
# Calculate failure rate for each producer
df['failure_rate'] = df['samples failed'] / df['samples taken']

# Find the producer with the lowest failure rate
lowest_failure_rate_producer = df.loc[df['failure_rate'].idxmin(), 'producer']
print(f"Final Answer: {lowest_failure_rate_producer}")