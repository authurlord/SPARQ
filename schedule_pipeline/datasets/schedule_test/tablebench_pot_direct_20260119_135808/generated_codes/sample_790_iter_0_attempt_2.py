import pandas as pd

df = pd.read_csv('table.csv')
# The 'average' column contains numerical values
std_average = df['average'].std()
print(f"Final Answer: {std_average:.2f}")