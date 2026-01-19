import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where 'just ratio' contains '11'
filtered_rows = df[df['just ratio'].str.contains('11', na=False)]

# Extract 'size (cents)' values
size_cents = filtered_rows['size (cents)'].astype(float)

# Calculate average
average_size = size_cents.mean()

print(f"Final Answer: {average_size:.2f}")