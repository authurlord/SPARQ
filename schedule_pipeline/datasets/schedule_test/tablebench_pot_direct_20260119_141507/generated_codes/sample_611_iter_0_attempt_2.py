import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where route is 'monte carlo ( mcm ) - nice ( nce )'
route_name = 'monte carlo ( mcm ) - nice ( nce )'
current_flights = df[df['route'] == route_name]['avg daily flts'].values[0]
# Increase by 28%
new_flights = current_flights * 1.28
print(f"Final Answer: {new_flights}")