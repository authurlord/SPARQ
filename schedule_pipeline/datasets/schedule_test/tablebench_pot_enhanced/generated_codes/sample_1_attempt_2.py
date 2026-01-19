import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'avg daily flts' column to integer type
df['avg daily flts'] = pd.to_numeric(df['avg daily flts'])
# Calculate total average daily flights
total_flights = df['avg daily flts'].sum()
print(f"Final Answer: {total_flights}")