import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where just ratio involves 11
filtered_df = df[df['just ratio'].isin(['11:8', '14:11', '11:9', '13:11'])]
# Calculate average size in cents
average_size_cents = filtered_df['size (cents)'].mean()
print(f"Final Answer: {average_size_cents:.2f}")