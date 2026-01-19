import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Glasgow International
glasgow_row = df[df['airport'] == 'glasgow international']
# Extract total passengers (convert from string to int)
total_passengers = int(glasgow_row['total passengers'].values[0])
# Apply 15% increase
projected_passengers = total_passengers * 1.15
print(f"Final Answer: {round(projected_passengers)}")