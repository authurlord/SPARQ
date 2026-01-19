import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'average' column to numeric to ensure proper arithmetic operations
df['average'] = pd.to_numeric(df['average'], errors='coerce')
# Get the average score of the top-ranked couple (first row)
top_average = df.iloc[0]['average']
# Get the average score of the bottom-ranked couple (last row)
bottom_average = df.iloc[-1]['average']
# Calculate the difference
difference = top_average - bottom_average
print(f"Final Answer: {difference:.1f}")