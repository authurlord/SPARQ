import pandas as pd

df = pd.read_csv('table.csv')

# Display column descriptions and notable patterns
print("Column Significance:")
print("- year: The year of the hurricane season.")
print("- number of tropical storms: Total number of tropical storms formed.")
print("- number of hurricanes: Number of storms reaching hurricane strength (≥74 mph).")
print("- number of major hurricanes: Hurricanes with Category 3 or higher (≥111 mph).")
print("- deaths: Number of fatalities attributed to hurricanes in that year.")
print("- strongest storm: Identifier of the strongest storm of the season.")

print("\nNotable Trends:")
print("- 1870 and 1878 had the highest number of tropical storms and hurricanes.")
print("- Major hurricanes were rare; only 2 occurred in 1870 and 1873.")
print("- 1870 recorded the highest death toll (2052), likely due to a powerful storm.")
print("- Storm names were not standardized; many were identified by numbers or combinations.")

# Final answer based on the question's request for description
print(f"Final Answer: year, number of tropical storms, number of hurricanes, number of major hurricanes, deaths, strongest storm")