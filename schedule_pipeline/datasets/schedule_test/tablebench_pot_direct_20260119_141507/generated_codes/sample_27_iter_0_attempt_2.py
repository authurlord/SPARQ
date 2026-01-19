import pandas as pd

df = pd.read_csv('table.csv')

# Filter drivers who drove a Chevrolet
chevrolet_drivers = df[df['make'] == 'chevrolet']

# Total winnings for Chevrolet drivers
total_winnings_chevrolet = chevrolet_drivers['winnings'].sum()

# Driver with the highest winnings among Chevrolet drivers
top_driver_chevrolet = chevrolet_drivers.loc[chevrolet_drivers['winnings'].idxmax(), 'driver']

print(f"Final Answer: {total_winnings_chevrolet}, {top_driver_chevrolet}")