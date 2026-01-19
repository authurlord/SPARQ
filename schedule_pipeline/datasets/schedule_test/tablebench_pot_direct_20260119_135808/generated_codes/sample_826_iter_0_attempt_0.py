import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the standard deviation of the 'total' column
std_total = df['total'].std()
print(f"Final Answer: {std_total:.2f}")