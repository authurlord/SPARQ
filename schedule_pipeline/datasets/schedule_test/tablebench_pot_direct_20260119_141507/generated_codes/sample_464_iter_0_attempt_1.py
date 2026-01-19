import pandas as pd

df = pd.read_csv('table.csv')

# Extract relevant columns
df = df.dropna()  # Remove any missing values

# Check if "Total size (bits)" matches sum of other fields
df['total_calc'] = (
    df['Sign field (bits)'] +
    df['Combination field (bits)'] +
    df['Exponent continuation field (bits)'] +
    df['Coefficient continuation field (bits)']
)

# Check deviation in total size
total_deviation = df[df['Total size (bits)'] != df['total_calc']].copy()

# Check coefficient size formula: p = 9×k − 2
# k is from 'decimal32', 'decimal64', 'decimal128'
df['k'] = df['decimal32'].apply(lambda x: int(x))
df['p_expected'] = 9 * df['k'] - 2
df['p_deviation'] = df['Coefficient size (decimal digits)'] != df['p_expected']

# Find rows where coefficient size deviates
coeff_deviation = df[df['p_deviation']].copy()

# Check exponent range: 3×2^w, w = 2×k + 4
df['w'] = 2 * df['k'] + 4
df['exp_range_expected'] = 3 * (2 ** df['w'])
df['exp_range_deviation'] = df['Exponent range'] != df['exp_range_expected']

# Find rows where exponent range deviates
exp_deviation = df[df['exp_range_deviation']].copy()

# Combine all deviations
deviations = total_deviation.append(coeff_deviation, ignore_index=True)
deviations = deviations.append(exp_deviation, ignore_index=True)

# Final answer: list the rows with deviations (by index or key)
if deviations.empty:
    print("Final Answer: no deviations")
else:
    print(f"Final Answer: {deviations[['decimal32', 'decimal64', 'decimal128']].to_string(index=False)}")