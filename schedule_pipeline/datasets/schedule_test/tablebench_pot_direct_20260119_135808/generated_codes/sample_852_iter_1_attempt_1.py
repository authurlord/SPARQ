import pandas as pd

df = pd.read_csv('table.csv')
# Convert the 'average' column to float for numerical operations
df['average'] = pd.to_numeric(df['average'], errors='coerce')
# Get the average score of the top-ranked couple (first row) and bottom-ranked couple (last row)
top_average = df['average'].iloc[0]
bottom_average = df['average'].iloc[-1]
# Calculate the difference
difference = top_average - bottom_average
print(f"Final Answer: {difference:.1f}")