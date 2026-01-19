import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Chevrolet cars (case-insensitive)
chevrolet_drivers = df[df['car'].str.lower() == 'chevrolet']
# Convert winnings to numeric
chevrolet_drivers['winnings'] = pd.to_numeric(chevrolet_drivers['winnings'], errors='coerce')
# Calculate total winnings
total_winnings = chevrolet_drivers['winnings'].sum()
# Find driver with highest winnings
top_driver = chevrolet_drivers.loc[chevrolet_drivers['winnings'].idxmax(), 'driver']
print(f"Final Answer: {total_winnings:.0f}, {top_driver}")