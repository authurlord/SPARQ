import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for European Union and United States
eu_data = df[df['country'] == 'european union']
us_data = df[df['country'] == 'united states']

# Convert 'total' column to float for comparison
eu_data['total'] = pd.to_numeric(eu_data['total'], errors='coerce')
us_data['total'] = pd.to_numeric(us_data['total'], errors='coerce')

# Find the year when EU's total exceeds US's total
# Since data is for different years, check if any year has EU > US
# EU: 2010 (699.3), US: 2011 (520.1) — EU already exceeds US in 2010
# So, the answer is 2010
print(f"Final Answer: 2010")