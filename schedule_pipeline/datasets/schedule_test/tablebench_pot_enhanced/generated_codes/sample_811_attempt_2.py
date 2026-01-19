import pandas as pd

df = pd.read_csv('table.csv')
# Calculate standard deviation of 'size (cents)'
std_cents = df['size (cents)'].std()
print(f"Final Answer: {std_cents:.2f}")