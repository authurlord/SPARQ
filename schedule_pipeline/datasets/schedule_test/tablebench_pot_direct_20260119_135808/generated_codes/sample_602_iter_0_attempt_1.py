import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Glasgow International airport
glasgow_row = df[df['airport'] == 'glasgow international']
# Extract total passengers and convert to integer
total_passengers = int(glasgow_row['total passengers'].values[0])
# Calculate 15% increase
projected_passengers = total_passengers * 1.15
print(f"Final Answer: {int(projected_passengers)}")