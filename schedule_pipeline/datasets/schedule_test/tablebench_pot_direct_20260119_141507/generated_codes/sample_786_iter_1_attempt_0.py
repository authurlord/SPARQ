import pandas as pd

df = pd.read_csv('table.csv')
# Filter drivers who drove a Chevrolet car
chevy_drivers = df[df['make'] == 'chevrolet']
# Remove rows where winnings is 'pe' and convert to numeric
chevy_drivers['winnings'] = pd.to_numeric(chevy_drivers['winnings'], errors='coerce')
# Drop any rows with NaN (invalid winnings)
chevy_drivers = chevy_drivers.dropna(subset=['winnings'])
# Calculate median of winnings
median_winnings = chevy_drivers['winnings'].median()
print(f"Final Answer: {median_winnings}")