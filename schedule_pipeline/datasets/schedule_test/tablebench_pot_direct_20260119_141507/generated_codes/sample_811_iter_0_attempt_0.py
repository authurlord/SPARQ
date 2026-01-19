import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the standard deviation of 'size (cents)'
std_size_cents = df['size (cents)'].std()
print(f"Final Answer: {std_size_cents:.2f}")