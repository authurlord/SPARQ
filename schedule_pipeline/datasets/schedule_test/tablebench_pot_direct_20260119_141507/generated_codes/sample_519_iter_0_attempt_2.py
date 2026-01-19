import pandas as pd

df = pd.read_csv('table.csv')
# Calculate fleet utilization rate: current destinations / passenger fleet
df['utilization_rate'] = df['current destinations'] / df['passenger fleet']

# Find the airline with the highest utilization rate
max_utilization_airline = df.loc[df['utilization_rate'].idxmax(), 'airline / holding']
print(f"Final Answer: {max_utilization_airline}")