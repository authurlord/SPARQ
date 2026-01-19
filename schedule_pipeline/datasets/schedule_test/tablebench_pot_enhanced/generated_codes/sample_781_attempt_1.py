import pandas as pd

df = pd.read_csv('table.csv')
# Calculate average number of major hurricanes per year
avg_major_hurricanes = df['number of major hurricanes'].mean()
print(f"Final Answer: {avg_major_hurricanes:.1f}")