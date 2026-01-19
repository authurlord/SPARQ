import pandas as pd

df = pd.read_csv('table.csv')

# Describe the main components and highlight notable differences
print("Main components of the crime statistics table:")
print("- Crime types (e.g., Murder, Robbery, Larceny – theft) are categorized.")
print("- Each row shows reported offenses and crime rates for Killeen, Texas, and U.S.")
print("\nNotable differences and trends:")
print("- Larceny – theft: Killeen (2,482.2) is lower than Texas (2,688.9), showing Texas has a higher rate.")
print("- Motor vehicle theft: Killeen (145.8) is significantly lower than Texas (351.1), indicating a major difference.")
print("- Burglary: Killeen (1,476.2) is higher than Texas (946.5), suggesting higher burglary rates in Killeen.")
print("- Violent crime: Killeen (763.5) is slightly above Texas (508.2) and U.S. (470.6), indicating elevated violent crime in Killeen.")
print("Final Answer: Larceny - theft, Motor vehicle theft, Burglary, Violent crime")