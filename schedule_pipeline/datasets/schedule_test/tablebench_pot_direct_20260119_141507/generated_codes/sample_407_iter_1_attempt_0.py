import pandas as pd

# Load the table
df = pd.read_csv('table.csv', index_col=0)

# Display the dataframe for clarity
print(df)

# Optional: Summary statistics for key columns
print("\nSummary of Key Metrics:")
print(f"Applications trend: {df['2017'].iloc[0]} (2017) → {df['2016'].iloc[0]} (2016)")
print(f"Offer Rate trend: {df['Offer Rate (%)'].iloc[0]:.1f} (2017) → {df['Offer Rate (%)'].iloc[1]:.1f} (2016)")
print(f"Yield trend: {df['Yield (%)'].iloc[0]:.1f} (2017) → {df['Yield (%)'].iloc[1]:.1f} (2016)")
print(f"Applicant/Enrolled Ratio: {df['Applicant/Enrolled Ratio'].iloc[0]:.2f} (2017) → {df['Applicant/Enrolled Ratio'].iloc[1]:.2f} (2016)")