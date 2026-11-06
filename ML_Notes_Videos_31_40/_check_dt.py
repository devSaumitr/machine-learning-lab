import pandas as pd
s = pd.Series(pd.to_datetime(["2019-12-12"]))
print("weekday attr:", hasattr(s.dt, "weekday"))
if hasattr(s.dt, "weekday"):
    print("weekday:", s.dt.weekday.tolist())
d = pd.Series(pd.to_timedelta(["40 days 3 hours"]))
print(d.dt.components)
