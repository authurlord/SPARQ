import pandas as pd

df = pd.read_csv('table.csv')
# Extract sales of the top 5 albums (first 5 rows)
top_5_sales = df.iloc[:5]['sales'].sum()
print(f"Final Answer: {top_5_sales}")