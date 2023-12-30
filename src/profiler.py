import pandas as pd


df = pd.read_csv("test.csv")
print(df.columns)
df = df.sort_values(by=["cumtime"], ascending=False)
#df = df[:10]
#df["filename:lineno(function)"] = [word.split(":")[2] for word in df["filename:lineno(function)"] if len(word.split(":")) >= 2]
import re
df["filename:lineno(function)"] = [re.sub("^.*[\\\\]","",word) for word in df["filename:lineno(function)"]]
print(df[:30])