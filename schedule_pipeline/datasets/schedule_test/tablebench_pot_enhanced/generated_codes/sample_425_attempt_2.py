import pandas as pd

df = pd.read_csv('table.csv')

# Display the first few rows for context
print("Table Overview:")
print(df.head())

# Column significance
print("\nColumn Significance:")
print("- year: The calendar year of the hurricane season.")
print("- number of tropical storms: Total tropical storms recorded.")
print("- number of hurricanes: Storms that reached hurricane status (≥74 mph).")
print("- number of major hurricanes: Hurricanes with Category 3 or higher (≥111 mph).")
print("- deaths: Number of fatalities attributed to storms; some values are 'unknown'.")
print("- strongest storm: Identifier of the most intense storm of the season.")

# Trends and patterns
print("\nNotable Trends and Patterns:")
print("- 1870 had the highest death toll (2052), likely due to a very strong storm ('four').")
print("- 1878 had the highest number of tropical storms (12) and hurricanes (10), indicating a very active season.")
print("- 1870 and 1873 both had 2 major hurricanes, but 1870 had significantly more deaths.")
print("- Years with 'unknown' deaths (e.g., 1872, 1874) suggest incomplete historical records.")
print("- The strongest storm names are often numerical (e.g., 'four', 'three and four'), suggesting a naming convention based on intensity or order.")

# Final summary
print("\nFinal Answer: year, number of tropical storms, number of hurricanes, number of major hurricanes, deaths, strongest storm")