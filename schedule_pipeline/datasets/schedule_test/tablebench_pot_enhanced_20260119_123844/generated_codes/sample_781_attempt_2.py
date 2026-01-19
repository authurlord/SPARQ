import pandas as pd

df = pd.read_csv('table.csv')
# Calculate average number of major hurricanes per year
avg_major_hurricanes_per_year = df['number of major hurricanes'].mean()
# Predict total for the next decade
predicted_total = avg_major_hurricanes_per_year * 10
print(f"Final Answer: {predicted_total:.1f}")