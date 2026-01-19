import pandas as pd

df = pd.read_csv('table.csv')
# Convert success rate to numeric by removing '%' and converting to float
df['success rate'] = df['success rate'].str.replace('%', '').astype(float)

# Get overall success rate from the 'total' row
overall_success_rate = df[df['year'] == 'total']['success rate'].values[0]

# Identify years with success rates more than 10% away from overall
threshold = 10.0
unusual_years = []
for index, row in df.iterrows():
    if row['year'] == 'total':
        continue
    diff = abs(row['success rate'] - overall_success_rate)
    if diff > threshold:
        unusual_years.append(row['year'])

print(f"Final Answer: {', '.join(unusual_years)}")