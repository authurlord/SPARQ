import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'number of hurricanes' column to integer to ensure proper comparison
df['number of hurricanes'] = pd.to_numeric(df['number of hurricanes'], errors='coerce')
# Count the number of years with 3 or fewer hurricanes
count_years = (df['number of hurricanes'] <= 3).sum()
print(f"Final Answer: {count_years}")