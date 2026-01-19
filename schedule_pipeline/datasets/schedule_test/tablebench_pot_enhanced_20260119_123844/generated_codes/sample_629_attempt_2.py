import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'sales' column to integer and sum the top 5
total_sales_top5 = df['sales'].astype(int).head(5).sum()
print(f"Final Answer: {total_sales_top5}")