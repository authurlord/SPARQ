import pandas as pd

df = pd.read_csv('table.csv')

# Descriptive explanation of the table
print("Main columns: rank, name, height feet (m), floors, year")
print("Insights:")
print("- Tallest building: One America Plaza (500 ft / 152 m)")
print("- Most floors: Electra (43 floors)")
print("- Earliest construction: 1927 (El Cortez Apartment Hotel)")
print("- Latest construction: 2012 (US Federal Courthouse)")
print("- Multiple buildings share the same rank (e.g., '5 =' and '8 =')")
Final Answer: rank, name, height feet (m), floors, year