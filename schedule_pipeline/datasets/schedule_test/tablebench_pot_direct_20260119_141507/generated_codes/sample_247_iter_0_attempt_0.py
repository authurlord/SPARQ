import pandas as pd

df = pd.read_csv('table.csv')
# Sort by 'weight lost (kg)' in descending order and take top 3
top_3_by_weight_lost = df.sort_values(by='weight lost (kg)', ascending=False).head(3)
# Calculate average of 'percentage lost' for these top 3
avg_percentage_lost = top_3_by_weight_lost['percentage lost'].mean()
print(f"Final Answer: {avg_percentage_lost:.2f}")