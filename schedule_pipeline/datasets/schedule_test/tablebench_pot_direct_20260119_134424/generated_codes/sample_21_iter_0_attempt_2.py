import pandas as pd

df = pd.read_csv('table.csv')
# Convert population column to integer and sum the top 5
total_population = df['population'].astype(int).head(5).sum()
print(f"Final Answer: {total_population}")