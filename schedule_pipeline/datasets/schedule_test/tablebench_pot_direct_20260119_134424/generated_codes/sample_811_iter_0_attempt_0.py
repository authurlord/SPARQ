import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'size (cents)' to numeric and calculate standard deviation
std_cents = df['size (cents)'].astype(float).std()
print(f"Final Answer: {std_cents:.2f}")