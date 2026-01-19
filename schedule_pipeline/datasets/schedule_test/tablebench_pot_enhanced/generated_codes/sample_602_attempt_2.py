import pandas as pd

df = pd.read_csv('table.csv')
# Find the row for Glasgow International
glasgow_row = df[df['airport'] == 'glasgow international']
# Extract the total passengers
total_passengers = int(glasgow_row['total passengers'].values[0])
# Calculate 15% increase
new_total = total_passengers * 1.15
print(f"Final Answer: {int(new_total)}")