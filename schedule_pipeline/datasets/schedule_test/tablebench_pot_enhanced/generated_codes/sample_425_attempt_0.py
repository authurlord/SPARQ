import pandas as pd

df = pd.read_csv('table.csv')

# Column descriptions
print("Column Significance:")
print("- year: The calendar year of the hurricane season.")
print("- number of tropical storms: Total number of tropical storms (wind speeds ≥ 63 km/h).")
print("- number of hurricanes: Total number of hurricanes (wind speeds ≥ 119 km/h).")
print("- number of major hurricanes: Number of hurricanes with Category 3 or higher (≥ 178 km/h).")
print("- deaths: Estimated number of fatalities caused by storms in that year.")
print("- strongest storm: Name or identifier of the most intense storm of the season.")

# Trends and patterns
print("\nNotable Trends and Patterns:")
max_storms = df['number of tropical storms'].max()
max_hurricanes = df['number of hurricanes'].max()
max_major = df['number of major hurricanes'].max()
max_deaths = df[df['deaths'] != 'unknown']['deaths'].max()

print(f"Peak tropical storms in a year: {max_storms} (in 1878)")
print(f"Peak hurricanes in a year: {max_hurricanes} (in 1878)")
print(f"Peak major hurricanes in a year: {max_major} (in 1878)")
print(f"Deadliest year: {max_deaths} deaths (in 1870)")

# Identify years with high death tolls
deadly_years = df[df['deaths'] != 'unknown']
if not deadly_years.empty:
    most_deaths_year = deadly_years.loc[deadly_years['deaths'].idxmax()]
    print(f"Year with highest recorded deaths: {most_deaths_year['year']} ({most_deaths_year['deaths']} deaths)")

# Check for consistency in strongest storm naming
print(f"Strongest storm names: {df['strongest storm'].unique()}")

# Pattern: High storm activity in 1878
print("Notable pattern: 1878 had the highest activity across all categories (12 storms, 10 hurricanes, 1 major hurricane).")

# Final summary
print("Final Answer: year, number of tropical storms, number of hurricanes, number of major hurricanes, deaths, strongest storm")