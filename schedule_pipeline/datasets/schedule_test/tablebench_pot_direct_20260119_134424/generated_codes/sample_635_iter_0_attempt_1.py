import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where just ratio contains '11'
filtered_df = df[df['just ratio'].str.contains('11', na=False)]

# Extract 'size (cents)' values and compute average
average_size_cents = filtered_df['size (cents)'].astype(float).mean()

print(f"Final Answer: {average_size_cents:.2f}")