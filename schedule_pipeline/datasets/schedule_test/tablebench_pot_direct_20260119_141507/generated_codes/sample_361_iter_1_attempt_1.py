import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'number of hurricanes' to integer
df['number of hurricanes'] = df['number of hurricanes'].astype(int)
# Count years with 3 or fewer hurricanes
years_with_3_or_fewer = df[df['number of hurricanes'] <= 3]['year'].count()
print(f"Final Answer: {years_with_3_or_fewer}")