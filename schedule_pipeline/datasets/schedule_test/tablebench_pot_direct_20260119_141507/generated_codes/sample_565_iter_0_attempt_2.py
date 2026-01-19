- For 1850: Year_1 = 1885 → value = 395,346
- For 1855: Year_1 = 1900 → value = 448,572
- For 1860: Year_1 = 1925 → value = 294,314
- For 1865: Year_1 = 1930 → value = 241,700
- For 1870: Year_1 = 1935 → value = 34,956
- Row 0: 1820 → Year_1 = 1885 → value = 395,346
- Row 2: 1830 → Year_1 = 1895 → value = 258,536
- Row 4: 1840 → Year_1 = 1905 → value = 1,026,499
- Row 6: 1850 → Year_1 = 1915 → value = 326,700
- Row 8: 1860 → Year_1 = 1925 → value = 294,314
- Row 10: 1870 → Year_1 = 1935 → value = 34,956
- For year 1850: value = 369,980
- For year 1855: value = 200,877 → decrease
- For year 1860: value = 153,640 → decrease
- For year 1865: value = 248,120 → increase
- For year 1870: value = 387,203 → increase
- 1855 - 1850: 200,877 - 369,980 = -169,103
- 1860 - 1855: 153,640 - 200,877 = -47,237
- 1865 - 1860: 248,120 - 153,640 = 94,480
- 1870 - 1865: 387,203 - 248,120 = 139,083
import pandas as pd
df = pd.read_csv('table.csv')
years_of_interest = df[df['Year'].between(1850, 1870)]
values = years_of_interest['Year'].map(lambda x: df.loc[df['Year'] == x, df.columns[0]].values[0])
values = df[df['Year'].between(1850, 1870)]['Year'].map(
    lambda x: df.loc[df['Year'] == x, df.columns[1]].values[0]
values = df[df['Year'].between(1850, 1870)].iloc[:, 1].astype(float)
diffs = values.diff().dropna()
max_diff = diffs.max()
max_year = df[df['Year'].between(1850, 1870)]['Year'].iloc[diffs.idxmax()]