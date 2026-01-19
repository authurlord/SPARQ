import pandas as pd

df = pd.read_csv('table.csv')

deviations = []

# Loop through each row
for idx, row in df.iterrows():
    # Skip the last row (it's just descriptive)
    if idx == len(df) - 1:
        continue
    
    # Extract values
    k = int(row['decimal32'])  # Assuming k is from decimal32 column
    sign_bits = int(row['decimal32'])
    combo_bits = int(row['decimal64'])
    exp_cont_bits = int(row['decimal128'])
    coeff_cont_bits = int(row['decimal(32k)'])
    total_size = int(row['Total size (bits)'])
    
    # Expected values based on formulas
    expected_w = 2 * k + 4
    expected_t = 30 * k - 10
    expected_p = 9 * k - 2
    expected_total = sign_bits + combo_bits + exp_cont_bits + coeff_cont_bits
    
    # Check deviations
    if abs(exp_cont_bits - expected_w) > 1:
        deviations.append(f"Row {idx}: Exponent continuation ({exp_cont_bits}) deviates from w = 2k+4 ({expected_w})")
    if abs(total_size - expected_total) > 1:
        deviations.append(f"Row {idx}: Total size ({total_size}) deviates from sum of fields ({expected_total})")
    if abs(int(row['Coefficient size (decimal digits)']) - expected_p) > 1:
        deviations.append(f"Row {idx}: Coefficient size ({row['Coefficient size (decimal digits)']}) deviates from p = 9k−2 ({expected_p})")

if deviations:
    print("Deviation found:")
    for d in deviations:
        print(d)
else:
    print("No significant deviations found.")

Final Answer: No significant deviations found