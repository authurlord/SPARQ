import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'total passengers' and 'capacity' columns
total_passengers = df['total passengers'].astype(float)
capacity = df['capacity'].astype(float)

# Calculate the correlation coefficient
correlation = total_passengers.corr(capacity)
print(f"Final Answer: {correlation:.2f}")