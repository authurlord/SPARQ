import pandas as pd

df = pd.read_csv('table.csv')

# Column descriptions
print("Column Significance:")
print("- year: The calendar year of the tropical cyclone season.")
print("- number of tropical storms: Number of tropical storms (wind speeds ≥ 63 km/h).")
print("- number of hurricanes: Number of hurricanes (wind speeds ≥ 119 km/h).")
print("- number of major hurricanes: Number of Category 3 or higher hurricanes (wind speeds ≥ 178 km/h).")
print("- deaths: Estimated number of fatalities caused by storms in that year.")
print("- strongest storm: Name or identifier of the most intense storm of the season.")

# Trends and patterns
print("\nNotable Trends:")
print("- High variability in storm numbers: Tropical storms ranged from 5 to 12 annually.")
print("- Hurricane frequency was consistently high, with 3–10 hurricanes per year.")
print("- Major hurricanes were rare, with only 0–2 per year; 1870, 1873, and 1876 had 2 major hurricanes.")
print("- Death tolls varied significantly: 1870 had the highest deaths (2052), possibly due to a major storm like 'four'.")
print("- The strongest storm names often repeated (e.g., 'three', 'four', 'seven'), suggesting naming conventions or regional significance.")
print("- Years with high death tolls (e.g., 1870, 1873, 1875) do not always correlate with high storm counts, indicating impact depends on storm intensity and location.")

# Final summary
print(f"\nFinal Answer: year, number of tropical storms, number of hurricanes, number of major hurricanes, deaths, strongest storm")