import pandas as pd

df = pd.read_csv('table.csv')
# Find the row for Glasgow International Airport
glasgow_row = df[df['airport'] == 'glasgow international']
# Extract total passengers
total_passengers_glasgow = int(glasgow_row['total passengers'].values[0])
# Calculate 15% increase
predicted_passengers_2008 = total_passengers_glasgow * 1.15
print(f"Final Answer: {round(predicted_passengers_2008)}")