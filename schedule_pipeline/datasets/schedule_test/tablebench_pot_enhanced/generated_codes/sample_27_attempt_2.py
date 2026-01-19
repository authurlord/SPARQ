import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'car' is 'chevrolet'
chevrolet_drivers = df[df['car'] == 'chevrolet']
# Calculate total winnings
total_winnings = chevrolet_drivers['winnings'].sum()
# Find the driver with the highest winnings
top_driver = chevrolet_drivers.loc[chevrolet_drivers['winnings'].idxmax(), 'driver']
print(f"Final Answer: {total_winnings}, {top_driver}")