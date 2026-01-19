import pandas as pd

df = pd.read_csv('table.csv')
# Filter out 'All Ages' and 'TOTAL' rows for analysis
data = df[~df['Age Group'].isin(['All Ages', 'TOTAL'])]
# Find the row with the highest percentage
max_row = data.loc[data['%'].idxmax()]
# Find the row with the lowest percentage
min_row = data.loc[data['%'].idxmin()]
# Calculate the difference
difference = float(max_row['%']) - float(min_row['%'])
# Output the result
print(f"Final Answer: {max_row['Age Group']}, {difference:.2f}")