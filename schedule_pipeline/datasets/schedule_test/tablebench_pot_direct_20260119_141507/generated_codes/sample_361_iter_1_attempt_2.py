import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'number of hurricanes' to integer type
df['number of hurricanes'] = pd.to_numeric(df['number of hurricanes'], errors='coerce')
# Filter years where number of hurricanes is 3 or fewer
years_with_3_or_fewer = df[df['number of hurricanes'] <= 3]
# Count the number of such years
count = len(years_with_3_or_fewer)
print(f"Final Answer: {count}")