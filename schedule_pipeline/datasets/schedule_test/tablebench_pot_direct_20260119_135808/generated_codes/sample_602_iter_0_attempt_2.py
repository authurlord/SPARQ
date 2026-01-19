import pandas as pd

df = pd.read_csv('table.csv')
# Find the row for Glasgow International
glasgow_row = df[df['airport'] == 'glasgow international']
# Extract total passengers (convert to integer)
total_passengers_glasgow = int(glasgow_row['total passengers'].values[0])
# Calculate 15% increase
predicted_passengers_2008 = total_passengers_glasgow * 1.15
print(f"Final Answer: {int(predicted_passengers_2008)}")