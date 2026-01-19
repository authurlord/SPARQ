import pandas as pd

df = pd.read_csv('table.csv')

# Convert area (km²) to numeric (in case of string formatting)
df['area (km 2 )'] = pd.to_numeric(df['area (km 2 )'], errors='coerce')

# Calculate correlation between population and area
correlation = df['population'].corr(df['area (km 2 )'])

# If correlation is above 0.3 or below -0.3, we consider it significant
if abs(correlation) >= 0.3:
    print(f"Final Answer: area (km 2 )")
else:
    print("Final Answer: no clear impact")