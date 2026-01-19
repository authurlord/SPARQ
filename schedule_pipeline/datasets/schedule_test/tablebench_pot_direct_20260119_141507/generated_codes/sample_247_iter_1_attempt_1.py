import pandas as pd

df = pd.read_csv('table.csv')
# Sort by weight lost (kg) in descending order and take top 3
top_3 = df.sort_values(by='weight lost (kg)', ascending=False).head(3)

# Extract percentage lost and remove '%' then convert to float
percentages = top_3['percentage lost'].str.replace('%', '', regex=False).astype(float)

# Calculate average
average_percentage = percentages.mean()
print(f"Final Answer: {average_percentage:.2f}")