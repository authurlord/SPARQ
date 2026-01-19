import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the average finish position over the years 1985 to 2004
avg_finish = df['avg finish'].mean()
print(f"Final Answer: {avg_finish:.1f}")