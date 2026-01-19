import pandas as pd

df = pd.read_csv('table.csv')
# Convert success rate to numeric by removing '%' and converting to float
df['success rate'] = df['success rate'].str.replace('%', '').astype(float)

# Calculate overall success rate from the total row
overall_success_rate = df[df['year'] == 'total']['success rate'].iloc[0]

# Define threshold for "unusually high or low" (e.g., more than 10 percentage points away)
threshold = 10.0

# Find years with success rate significantly different from overall
unusual_years = df[
    (df['year'] != 'total') &
    ((df['success rate'] - overall_success_rate) > threshold |
     (df['success rate'] - overall_success_rate) < -threshold)
]['year'].tolist()

print(f"Final Answer: {', '.join(unusual_years)}")