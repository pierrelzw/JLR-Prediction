import re
from scipy import stats
import os
import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np  # fundamental package for acientific computing with python
import matplotlib 
from matplotlib import pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import config as cfg

from plotly import tools

init_notebook_mode(connected=True)



#----------------------------------------
# data cleaning on dataframe level
#----------------------------------------
def clean_df(df):
    print("Original shape: ", df.shape)
    # replace sting '[]' to NAN
    df.replace('[]', np.nan, inplace=True)

    # remove duplicate rows
    
    df.drop_duplicates(inplace=True)
    print("After removing duplication: ", df.shape)
    
    # remove empty rows
    df.dropna(how='all',axis=0, inplace=True)
    print("After removing empty rows: ", df.shape)

    # remove empty columns
    df.dropna(how='all',axis=1, inplace=True)
    print("After removing empty cols: ",df.shape)


#----------------------------------------
# feature extraction
#----------------------------------------
# post 帖子
def extract_excellent_post_count(row):
    return int(re.findall("\d+",row["post_count"].split("/")[0])[0])

def extract_post_count(row):
    return int(re.findall("\d+",row["post_count"].split("/")[1])[0])

# 关注的车
def extract_car_info(row):
    s = row["car_liked"].strip().split("：")
    if len(s)>1:
        return s[1]
    else:
        return ''

def gender2bool(row):
    return cfg._C.dict_gender[row['gender']] + 1

def loc2value(row):
    locs = cfg._C.LOCATIONS 
    len_loc = len(locs)
    dict_loc = dict(zip(locs, range(len_loc)))
    return dict_loc[row['location']] + 1


def extract_year(row):
    return row['regis_year'].dt.year.astype('uint16')
def extract_month(row):
    return row['regis_month'].dt.month.astype('uint8')
def extract_day(row):
    return row['regis_day'].dt.day.astype('uint8')

def property2value(row):
    properties = cfg._C.PROPERTIES
    len_prop = len(properties)
    dict_prop = dict(zip(properties, range(len_prop)))
    return dict_prop[row['properties']] + 1
    
def cars_like2value(row):
    import config as cfg
    cars = cfg._C.CARS_LIKE
    len_cars = len(cars)
    dict_cars = dict(zip(cars, range(len_cars)))
    if str(row['car_like']) == 'nan':
        return 0
    else:
        return dict_cars[row['car_like']] + 1 
    
#----------------------------------------
# data exploration
#----------------------------------------

# check missing data
def missing_data_check(df):
    print("dataset size: ",df.shape)
    total = df.isnull().sum().sort_values(ascending=False)
    percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)
    missing_data = pd.concat([total,percentage], axis=1,keys=['total', 'missing_percentage'])
    print(missing_data.head(18))
    

# The function to plot the distribution of the categorical values Horizontaly 
def bar_hor(df,  col, title, color, w=None, h=None, lm=0,  limit=100, return_trace=False, rev=False, xlb=False):
    cnt_srs = df[col].value_counts()
    yy = cnt_srs.head(limit).index[::-1]
    xx = cnt_srs.head(limit).values[::-1]
    if rev:
        yy = cnt_srs.tail(limit).index[::-1]
        xx = cnt_srs.tail(limit).values[::-1]
    if xlb:#????
        trace = go.Bar(y=xlb, x=xx,orientation='h', marker=dict(color=color))
    else:
        trace = go.Bar(y=yy, x=xx,orientation='h', marker=dict(color=color))
    if return_trace:
        return trace
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)

    
# The function to get the distribution of the categories according to the target
#(target de dtype=bool? or np.int8?)
def gp(df, col, title):
    df0 = df[df['label']==0]
    df1 = df[df['label']==1]
    a1 = df0[col].value_counts()
    b1 = df1[col].value_counts()
    
    total = dict(df[col].value_counts())
    x0 = a1.index
    x1 = b1.index
    
    y0 = [float(x)*100/total[x0[i]] for i,x in enumerate(a1.values)]
    y1 = [float(x)*100/total[x1[i]] for i,x in enumerate(b1.values)]
    
#     y0, x0 = zip(*sorted(zip(y0, x0), reverse=True))
#     y1, x1 = zip(*sorted(zip(y1, x1), reverse=True))
    
    trace1 = go.Bar(x=x0, y=y0, name="Target : 0", marker=dict(color="#96D38C"),hoverinfo="all")
    trace2 = go.Bar(x=x1, y=y1, name="Target : 1", marker=dict(color="#FEBFB3"),hoverinfo="all")
    
    
    return trace1, trace2

def exploreCat(df, col):
    t = df[col].value_counts()
    labels = t.index 
    values = t.values
    colors = ["#96D38C",  "#FEBFB3"]
    trace  = go.Pie(labels=labels, values=values,
                   hoverinfo="all",textinfo='value',
                   textfont=dict(size=12), 
                   marker=dict(colors=colors,
                               line=dict(color='#fff',width=2)))
    layout = go.Layout(title=col, height=400)
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)

# the relation between the categorical column and the target
def catAndTrgt(df, col, limit=100):
    tr0 = bar_hor(df, col, "Distribution of " + col, "#f975ae", w=700, lm=100, limit=limit, return_trace=True)
    tr1, tr2 = gp(df, col, "Distribution of Target with "+col)
    
    fig = tools.make_subplots(rows=1, cols=3, print_grid=False, 
                             subplot_titles=[col+" Distribution", "% of target=0", "% of target=1"])
    fig.append_trace(tr0, 1, 1);
    fig.append_trace(tr1, 1, 2);
    fig.append_trace(tr2, 1, 3);
    fig['layout'].update(height=350, showlegend=False, margin=dict(l=50));
    iplot(fig);

