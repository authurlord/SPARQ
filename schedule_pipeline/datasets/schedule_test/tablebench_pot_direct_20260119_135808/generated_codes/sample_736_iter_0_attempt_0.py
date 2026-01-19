import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for Suzuki in 50cc and 125cc classes
suzuki_data = df[(df['team'] == 'suzuki') & (df['class'].isin(['50cc', '125cc']))]
# Calculate average points for each class
avg_points_50cc = suzuki_data[suzuki_data['class'] == '50cc']['points'].mean()
avg_points_125cc = suzuki_data[suzuki_data['class'] == '125cc']['points'].mean()
print(f"Final Answer: {avg_points_50cc:.1f}, {avg_points_125cc:.1f}")