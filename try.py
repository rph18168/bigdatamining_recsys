import pandas as pd

data = {
    'aa': [1, 1, 3],
    'bb': [4, 5, 6],
    'cc': [7, 8, 9]
}
df = pd.DataFrame(data)
print(df)

a = df[df['aa'] == 1]['cc']
b = df['bb'][2]
print(b)
print(type(b))
print(a.iloc[0])
print(type(a))
