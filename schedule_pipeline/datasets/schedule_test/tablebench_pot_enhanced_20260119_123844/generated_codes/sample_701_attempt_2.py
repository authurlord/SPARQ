import pandas as pd

df = pd.read_csv('table.csv')
# Clean the 'Injuries (US $000)' column: remove non-numeric characters and convert to float
df['Injuries (US $000)'] = df['Injuries (US $000)'].astype(str).str.replace(r'[^\d.]', '', regex=True)
df['Injuries (US $000)'] = pd.to_numeric(df['Injuries (US $000)'], errors='coerce')

# Find the year with the maximum injuries
max_injuries_year = df.loc[df['Injuries (US $000)'].idxmax(), 'Year']
print(f"Final Answer: {max_injuries_year}")