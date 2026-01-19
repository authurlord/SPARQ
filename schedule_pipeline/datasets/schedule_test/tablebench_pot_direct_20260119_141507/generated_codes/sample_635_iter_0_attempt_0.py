import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where 'just ratio' contains '11'
filtered_rows = df[df['just ratio'].str.contains('11', na=False)]

# Calculate the average of 'size (cents)' for these filtered rows
average_cents = filtered_rows['size (cents)'].mean()

print(f"Final Answer: {average_cents:.2f}")