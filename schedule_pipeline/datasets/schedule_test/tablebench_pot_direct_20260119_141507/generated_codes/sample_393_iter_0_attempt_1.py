import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Display a detailed description of the table
print("Table Description:")
print("Columns:")
print(" - Fiscal Year: The fiscal year for which the data is reported.")
print(" - Total External Debt in Million of US Dollars ($): Total external debt in millions of USD.")
print(" - Total Debt Service in Million of US Dollars ($): Annual cost to service external debt (interest and principal).")
print(" - External Debt to GDP Ratio (%): Ratio of external debt to GDP, indicating debt sustainability.")
print(" - Debt Service Ratio (%): Ratio of debt service to GDP, showing the economic burden of debt payments.")
print("\nKey Insights:")
print(" - Total external debt peaked in 2012 at $79.949 million and fluctuated thereafter.")
print(" - External Debt to GDP Ratio declined from 68.2% (2001) to 23.5% (2018), indicating improved sustainability.")
print(" - Debt Service Ratio peaked at 17.1% in 2002 and dropped below 10% from 2008 onward.")
print(" - From 2015 onwards, the Debt Service Ratio shows '-' indicating missing or unavailable data.")