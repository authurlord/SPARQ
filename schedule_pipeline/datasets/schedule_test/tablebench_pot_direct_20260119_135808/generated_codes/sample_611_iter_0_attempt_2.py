import pandas as pd

df = pd.read_csv('table.csv')
# Filter the specific route
route_row = df[df['route'] == 'monte carlo ( mcm ) - nice ( nce )']
# Extract the current average daily flights
current_flights = int(route_row['avg daily flts'].values[0])
# Calculate 28% increase
new_flights = current_flights * 1.28
print(f"Final Answer: {new_flights:.0f}")