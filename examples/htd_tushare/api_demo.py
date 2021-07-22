import os 
import tushare as ts 

ts.set_token(os.environ.get('TUSHARE_TOKEN', None))

print(ts.__version__)

pro = ts.pro_api()

df = pro.news(src='sina', start_date='2020-01-01', end_date='2020-01-02')
print(df)