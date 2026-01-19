import pandas as pd

df = pd.read_csv('table.csv')

# Sort by weight lost (kg) in descending order and take top 3
top_3 = df.sort_values(by='weight lost (kg)', ascending=False).head(3)

# Remove '%' and convert to float
percentage_lost_numeric = top_3['percentage lost'].str.replace('%', '').astype(float)

# Calculate average
avg_percentage = percentage_lost_numeric.mean()

print(f"Final Answer: {avg_percentage:.2f}")