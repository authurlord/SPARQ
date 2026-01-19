import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'avg daily flts' column to integers
df['avg daily flts'] = df['avg daily flts'].astype(int)
# Calculate total average daily flights
total_flights = df['avg daily flts'].sum()
print(f"Final Answer: {total_flights}")