import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where just ratio contains '11'
filtered_df = df[df['just ratio'].str.contains('11', na=False)]
# Calculate average size in cents
average_cents = filtered_df['size (cents)'].astype(float).mean()
print(f"Final Answer: {average_cents:.2f}")