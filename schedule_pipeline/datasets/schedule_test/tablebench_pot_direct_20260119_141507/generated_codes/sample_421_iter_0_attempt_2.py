import pandas as pd

df = pd.read_csv('table.csv')

# Describe the main components and trends
print("Main components of the table:")
print("- 'department': Administrative regions in Bolivia.")
print("- 'micro (10ha)': Land area ≤10 hectares.")
print("- 'small (100ha)': Land area between 100 and 500 hectares.")
print("- 'medium (500ha)': Land area between 500 and 500 hectares (likely a typo).")
print("- 'big (>500ha)': Land area greater than 500 hectares.")
print("- 'total': Total land area per department.")

print("\nNotable trends:")
print("- Cochabamba has the highest total land area, dominated by small and big land categories.")
print("- La Paz has substantial land in small and medium categories.")
print("- Oruro and Santa Cruz have lower totals, with Oruro having a high proportion of large land holdings.")
print("- The 'big (>500ha)' category is rare, suggesting limited large-scale land ownership.")