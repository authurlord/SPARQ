import pandas as pd

df = pd.read_csv('table.csv')
# Get the average score of the top-ranked couple (first row)
top_average = df['average'].iloc[0]
# Get the average score of the bottom-ranked couple (last row)
bottom_average = df['average'].iloc[-1]
# Calculate the difference
difference = top_average - bottom_average
print(f"Final Answer: {difference:.1f}")