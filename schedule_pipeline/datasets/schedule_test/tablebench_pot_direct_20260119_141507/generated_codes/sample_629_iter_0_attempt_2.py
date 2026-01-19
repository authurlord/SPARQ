import pandas as pd

df = pd.read_csv('table.csv')
# Get the sales of the top 5 albums (first 5 rows)
top_5_sales = df.head(5)['sales'].sum()
print(f"Final Answer: {top_5_sales}")