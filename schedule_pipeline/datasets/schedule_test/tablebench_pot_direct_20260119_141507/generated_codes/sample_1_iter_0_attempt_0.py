import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'avg daily flts' to integer and sum
total_flights = df['avg daily flts'].astype(int).sum()
print(f"Final Answer: {total_flights}")