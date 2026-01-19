import pandas as pd

df = pd.read_csv('table.csv')

# Display the table structure and description
print("Table Description:")
print("The table shows average annual growth rates (in %) for various regions/countries across four time periods: 1985–1990, 1990–1995, 1995–2000, and 2000–2005.")
print("Columns '1985 - 1990' to '2000 - 2005' represent growth rates during these periods.")
print()

print("Key Observations:")
print("- Asia and its subregions (China, East Asia, South-East Asia) show a consistent decline in growth rates over time.")
print("- China had the highest growth rates initially (5.04% in 1985–1990), but declined significantly by 2000–2005 (3.08%).")
print("- Europe experienced a sharp decline in growth rates, from 0.78% to 0.13%.")
print("- Oceania maintained relatively stable growth (~1.5%) throughout the periods.")
print("- North America showed fluctuation: dropped from 1.24% to 0.57%, then rose to 1.51% and slightly fell to 1.37%.")
print()

# Highlight notable trends
print("Notable Trends:")
print("1. Overall global trend: Declining growth rates in most regions after the 1990s.")
print("2. China's high initial growth followed by stabilization.")
print("3. Europe’s near stagnation in recent years.")
print("4. North America’s recovery in the 2000–2005 period.")

print(f"Final Answer: Asia, South-East Asia, East Asia, China, Europe, North America, Oceania")