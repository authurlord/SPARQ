import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'car' is 'chevrolet'
chevrolet_drivers = df[df['car'] == 'chevrolet']
# Convert winnings to numeric and calculate median
median_winnings = chevrolet_drivers['winnings'].astype(float).median()
print(f"Final Answer: {median_winnings:.0f}")