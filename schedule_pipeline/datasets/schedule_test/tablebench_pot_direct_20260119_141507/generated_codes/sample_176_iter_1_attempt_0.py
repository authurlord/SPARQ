import pandas as pd

df = pd.read_csv('table.csv')

# Select only numerical columns for correlation
numeric_columns = df.select_dtypes(include=['number'])
# Compute correlation with 'viewers (in millions)'
correlation_with_viewers = df['viewers (in millions)'].corr(df['rank'])

# Check if the correlation is significant (absolute value > 0.3)
if abs(correlation_with_viewers) > 0.3:
    # Rank has a moderate to strong influence
    final_answer = "rank"
else:
    final_answer = "no clear impact"

print(f"Final Answer: {final_answer}")