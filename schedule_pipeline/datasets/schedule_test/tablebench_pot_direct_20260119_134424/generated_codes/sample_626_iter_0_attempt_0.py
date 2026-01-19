import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'Year built' is 1909
locomotives_1909 = df[df['Year built'] == '1909']
# Count the number of locomotives
total_1909 = len(locomotives_1909)
print(f"Final Answer: {total_1909}")