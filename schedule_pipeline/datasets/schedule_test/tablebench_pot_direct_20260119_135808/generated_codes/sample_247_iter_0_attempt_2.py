import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'weight lost (kg)' to numeric for sorting
df['weight lost (kg)'] = pd.to_numeric(df['weight lost (kg)'])
# Sort by weight lost in descending order and get top 3
top_3 = df.nlargest(3, 'weight lost (kg)')
# Calculate average percentage lost for these 3
avg_percentage_lost = top_3['percentage lost'].mean()
print(f"Final Answer: {avg_percentage_lost:.2f}")