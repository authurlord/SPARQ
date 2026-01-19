import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'make' is 'chevrolet'
chevrolet_drivers = df[df['make'].str.lower() == 'chevrolet']
# Convert 'winnings' to numeric, handling any non-numeric values
chevrolet_drivers['winnings'] = pd.to_numeric(chevrolet_drivers['winnings'], errors='coerce')
# Drop rows with NaN winnings
chevrolet_drivers = chevrolet_drivers.dropna(subset=['winnings'])
# Calculate median winnings
median_winnings = chevrolet_drivers['winnings'].median()
print(f"Final Answer: {median_winnings:.0f}")