import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Clean and convert columns to numeric
df['total viewers'] = df['total viewers'].str.replace(',', '').astype(int)
df['share'] = df['share'].str.replace('%', '').astype(float)
df['bbc one weekly ranking'] = df['bbc one weekly ranking'].astype(int)

# Calculate correlation matrix
correlation_matrix = df[['total viewers', 'share', 'bbc one weekly ranking']].corr()

# Extract correlation between each factor and ranking
viewers_corr = correlation_matrix['bbc one weekly ranking']['total viewers']
share_corr = correlation_matrix['bbc one weekly ranking']['share']

# Check for significant influence (absolute correlation > 0.5)
if abs(viewers_corr) > 0.5 or abs(share_corr) > 0.5:
    if abs(viewers_corr) > abs(share_corr):
        print("Final Answer: total viewers")
    else:
        print("Final Answer: share")
else:
    print("Final Answer: no clear impact")