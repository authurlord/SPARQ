import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the standard deviation of the 'average' column
std_average = df['average'].std()
print(f"Final Answer: {std_average:.2f}")