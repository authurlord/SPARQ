import pandas as pd

df = pd.read_csv('table.csv')

# Filter drivers who drove a Chevrolet
chevrolet_drivers = df[df['make'] == 'chevrolet']

# Total winnings for Chevrolet drivers
total_winnings = chevrolet_drivers['winnings'].sum()

# Driver with highest winnings among Chevrolet drivers
top_driver = chevrolet_drivers.loc[chevrolet_drivers['winnings'].idxmax(), 'driver']

print(f"Final Answer: {total_winnings}, {top_driver}")