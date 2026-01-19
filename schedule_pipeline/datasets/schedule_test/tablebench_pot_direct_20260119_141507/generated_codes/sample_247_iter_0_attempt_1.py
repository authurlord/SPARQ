import pandas as pd

df = pd.read_csv('table.csv')
# Sort by weight lost (kg) in descending order and take top 3
top_3 = df.sort_values(by='weight lost (kg)', ascending=False).head(3)
# Extract percentage lost for top 3
percentages = top_3['percentage lost'].astype(float)
average_percentage = percentages.mean()
print(f"Final Answer: {average_percentage:.2f}")