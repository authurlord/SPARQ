import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Filter drivers who drove a Chevrolet car
chevy_drivers = df[df['make'] == 'chevrolet']
# Convert 'winnings' to numeric, coercing errors to NaN
chevy_drivers['winnings'] = pd.to_numeric(chevy_drivers['winnings'], errors='coerce')
# Calculate the median of winnings
median_winnings = chevy_drivers['winnings'].median()
print(f"Final Answer: {median_winnings:.0f}")