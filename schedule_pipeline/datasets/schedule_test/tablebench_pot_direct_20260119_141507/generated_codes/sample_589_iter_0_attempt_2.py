import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where year of marriage is 1846 and sum the number of children
total_children_1846 = df[df['year of marriage'] == '1846']['of children'].sum()
print(f"Final Answer: {total_children_1846}")