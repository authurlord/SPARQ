import pandas as pd

df = pd.read_csv('table.csv')
# Find the row for Glasgow International
glasgow_passengers = df[df['airport'] == 'glasgow international']['total passengers'].values[0]
# Calculate 15% increase
new_passengers = glasgow_passengers * 1.15
print(f"Final Answer: {int(new_passengers)}")