import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Convert sales and market value to numeric, handling potential formatting issues
df['sales (billion )'] = pd.to_numeric(df['sales (billion )'], errors='coerce')
df['market value (billion )'] = pd.to_numeric(df['market value (billion )'], errors='coerce')

# Filter oil and gas companies with sales >= 300 billion
filtered_df = df[(df['industry'] == 'oil and gas') & (df['sales (billion )'] >= 300)]

# Calculate average market value of filtered companies
if not filtered_df.empty:
    avg_market_value = filtered_df['market value (billion )'].mean()
    print(f"Final Answer: {avg_market_value:.1f}")
else:
    print("Final Answer: 0")