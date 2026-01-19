import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the mean of 'males (%)' column
avg_males_percentage = df['males (%)'].mean()
print(f"Final Answer: {avg_males_percentage:.2f}")