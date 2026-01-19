import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'average' column
averages = df['average'].astype(float)
# Top-ranked couple (rank 1) is at index 0, bottom-ranked (rank 12) at index 11
top_avg = averages.iloc[0]
bottom_avg = averages.iloc[-1]
difference = top_avg - bottom_avg
print(f"Final Answer: {difference:.1f}")