import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where route is 'monte carlo ( mcm ) - nice ( nce )'
route_row = df[df['route'] == 'monte carlo ( mcm ) - nice ( nce )']
# Extract the average daily flights and convert to float
avg_flights = float(route_row['avg daily flts'].values[0])
# Increase by 28%
new_avg_flights = avg_flights * 1.28
print(f"Final Answer: {new_avg_flights:.0f}")