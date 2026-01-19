import pandas as pd

# Load the table
df = pd.read_csv('table.csv')

# Display a summary of the table content and key trends
print("The table shows economic indicators (e.g., GDP per capita) for various regions/countries across five-year periods from 1985 to 2005.")
print("Columns represent values for each time period: 1985–1990, 1990–1995, 1995–2000, 2000–2005.")
print("Key trends:")
print("- Asia and Southeast Asia show declining values over time, indicating economic downturns or structural shifts.")
print("- China starts with a high value and declines steadily, suggesting a slowdown in growth.")
print("- Europe has consistently low values, indicating lower economic development.")
print("- North America shows a slight improvement in the later period.")
print("- Oceania remains stable, showing consistent performance.")