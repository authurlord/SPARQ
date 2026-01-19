import pandas as pd

df = pd.read_csv('table.csv')

# Sort by 'weight lost' in descending order and take top 3
top_3 = df.sort_values(by='weight lost', ascending=False).head(3)

# Extract percentage lost (remove '%' and convert to float)
percentage_lost = top_3['percentage lost'].str.replace('%', '', regex=False).astype(float)

# Calculate average
average_percentage = percentage_lost.mean()

print(f"Final Answer: {average_percentage:.2f}")