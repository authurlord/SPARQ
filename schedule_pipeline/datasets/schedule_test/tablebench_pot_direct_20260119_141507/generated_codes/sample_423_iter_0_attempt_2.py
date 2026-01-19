import pandas as pd

df = pd.read_csv('table.csv')

# Display the structure and basic info
print("Main Columns:")
print(" - city / municipality: Names of the cities or municipalities.")
print(" - no of barangays: Number of administrative divisions in each city.")
print(" - area (km square): Geographic area of each municipality in square kilometers.")
print(" - population (2010 census): Population recorded in the 2010 census.")
print(" - pop density (per km square): Population per square kilometer, calculated as population divided by area.")

# Initial insights
print("\nInitial Insights:")
print("- Taytay has the highest population density (7447.32), indicating a very densely populated area.")
print("- Angono and San Mateo have relatively low areas but high population densities.")
print("- Binangonan and Tanay have large populations and moderate to high densities.")
print("- Antipolo and Rodriguez have large areas and high populations, suggesting more spread-out settlements.")