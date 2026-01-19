import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'sales' column to integer
df['sales'] = df['sales'].astype(int)
# Sum the sales of the top 5 albums
total_sales = df['sales'].head(5).sum()
print(f"Final Answer: {total_sales}")