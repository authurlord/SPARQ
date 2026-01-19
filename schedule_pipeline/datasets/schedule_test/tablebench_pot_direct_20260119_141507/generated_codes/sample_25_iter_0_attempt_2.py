import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the mean of 'males (%)' column
average_males_percent = df['males (%)'].mean()
print(f"Final Answer: {average_males_percent:.1f}")