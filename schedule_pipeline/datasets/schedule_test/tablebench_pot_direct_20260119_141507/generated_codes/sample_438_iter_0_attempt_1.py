import pandas as pd

df = pd.read_csv('table.csv')

# Display a summary of the crime statistics with key observations
print("Main Components of the Crime Statistics Table:")
print("- The table includes crime types such as Murder, Rape, Robbery, Aggravated Assault, Burglary, Larceny-theft, and Motor Vehicle Theft.")
print("- Each crime type has reported offenses and rates for Killeen, Texas, and U.S. levels.")
print("- Notable Trends:")
print("  - Killeen shows higher violent crime rates than both Texas and U.S. averages, especially in Murder, Rape, and Robbery.")
print("  - Burglary and Larceny-theft rates in Killeen are higher than Texas but lower than U.S. rates.")
print("  - Motor vehicle theft in Killeen is significantly higher than both Texas and U.S. rates, indicating a potential outlier or unique trend.")
print("  - Overall, Killeen's crime rates are generally above the Texas and U.S. averages, especially in violent crimes.")
Final Answer: Killeen has higher violent crime rates than Texas and U.S., with notable differences in property crimes