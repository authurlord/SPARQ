import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where just ratio contains '11'
filtered_df = df[df['just ratio'].str.contains('11', na=False)]
# Extract 'size (cents)' and calculate average
average_size = filtered_df['size (cents)'].astype(float).mean()
print(f"Final Answer: {average_size:.2f}")