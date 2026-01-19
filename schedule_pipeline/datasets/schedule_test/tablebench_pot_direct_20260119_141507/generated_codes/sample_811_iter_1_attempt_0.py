import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'size (cents)' to float and compute standard deviation
size_cents_float = df['size (cents)'].str.replace(',', '').astype(float)
std_size_cents = size_cents_float.std()
print(f"Final Answer: {std_size_cents:.2f}")