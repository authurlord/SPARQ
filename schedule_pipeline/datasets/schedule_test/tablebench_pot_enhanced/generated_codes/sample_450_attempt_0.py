import pandas as pd

df = pd.read_csv('table.csv')
# Convert success rate to float by removing % and converting to decimal
df['success rate'] = df['success rate'].str.replace('%', '').astype(float)
# Get overall success rate from total row
overall_success_rate = df[df['year'] == 'total']['success rate'].values[0]
# Define threshold for "unusually high or low" (e.g., more than 10% deviation)
threshold = 10.0
# Find years with success rate deviating more than threshold from overall
outliers = df[df['year'] != 'total']
unusual_years = outliers[
    (abs(outliers['success rate'] - overall_success_rate) > threshold)
]['year'].tolist()
print(f"Final Answer: {', '.join(unusual_years)}")