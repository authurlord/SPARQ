import pandas as pd

df = pd.read_csv('table.csv')

# Select numerical columns
numerical_cols = ['elevation (m)', 'prominence (m)', 'col (m)']

# Function to detect outliers using IQR
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)

# Apply outlier detection to each column
outlier_rows = []
for idx, row in df.iterrows():
    if (detect_outliers_iqr(df['elevation (m)']).loc[idx]) or \
       (detect_outliers_iqr(df['prominence (m)']).loc[idx]) or \
       (detect_outliers_iqr(df['col (m)']).loc[idx]):
        outlier_rows.append(row.tolist())

# If no outliers found, return message
if not outlier_rows:
    print("Final Answer: no outliers")
else:
    print("Final Answer:", ", ".join([f"{row[1]} ({row[2]}, {row[3]})" for row in outlier_rows]))