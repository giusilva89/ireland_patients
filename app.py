# Import libraries
import pandas as pd
import numpy as np
import collections

# Static data visualisation
import matplotlib.pyplot as plt
import seaborn as sns


# Import Plotly libraries
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.offline as py 
from plotly.subplots import make_subplots

# Dash Libraries
import dash
from dash import html
from dash import Dash, Input, Output, callback
from dash.dash_table.Format import Group
import dash_bootstrap_components as dbc
from dash import dcc
from dash.dependencies import Input, Output
import dash_daq as daq

# Import library to read from URL using request
from urllib.request import urlopen
import urllib.request
import random

# Read json files
import json


# Scipy
import scipy 
from statsmodels.stats.weightstats import ztest
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from scipy.stats import shapiro
from scipy.stats import anderson
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from scipy.stats import kruskal
from scipy.stats import friedmanchisquare
import scipy.stats as stats
import scipy as sy


#Mann-Kendall Test
import pymannkendall as mk

# Stats libraries
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.formula.api import ols
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.seasonal import seasonal_decompose


# Time series models
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm_notebook

# Itertools
from itertools import product
import itertools

# Importing metrics
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import ssl

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")


# Create app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])

# Adjust app layout for mobile devices
meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=0.86, maximum-scale=5.0, minimum-scale=0.86"}]

# Create the app server
server = app.server

def read_geojson(url):
    # Bypass SSL Verification
    ssl._create_default_https_context = ssl._create_unverified_context

    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode())
    return data

# URL
irish_url = 'https://gist.githubusercontent.com/pnewall/9a122c05ba2865c3a58f15008548fbbd/raw/5bb4f84d918b871ee0e8b99f60dde976bb711d7c/ireland_counties.geojson'

# Read geojson file
jdata = read_geojson(irish_url)

# Access the features key
thelist = jdata['features']

# Loop through the to get the Counties id
locations =  [item['id'] for item in thelist ]

# Read the data
df = pd.read_csv('DHA41.20220216T120205.csv')

# Transform Year variable into string type
df['Year'] = df['Year'].astype(str)

# Filter out to avoid redundancy
df = df[
    (df['Age Group'] != 'All ages') &
    (df['Sex']!= 'Both sexes') &
    (df['Area'] != 'Ireland')
       ]

df = df[df['VALUE']>=0]
       


# Fill missing values
df = df.interpolate(method ='linear', limit_direction ='forward')

# Filter out UNIT for only Numbers
df = df.query("UNIT=='Number'")
# Filter out
df = df[df['Area'] != 'Ireland excl. non-residents']

# Rename Counties for the Data Visualisation section
df['Area'] = df["Area"].str.replace('Dublin City and County','Dublin',regex=False)
df['Area'] = df['Area'].str.replace('North Tipperary','Tipperary',regex=False)
df['Area'] = df['Area'].str.replace('South Tipperary','Tipperary',regex=False)


# ANOVA Function
def welch_anova_np(*args, var_equal=False):
    # https://svn.r-project.org/R/trunk/src/library/stats/R/oneway.test.R
    # translated from R Welch ANOVA (not assuming equal variance)

    F_onewayResult = collections.namedtuple('F_onewayResult', ('statistic', 'pvalue'))

    args = [np.asarray(arg, dtype=float) for arg in args]
    k = len(args)
    ni =np.array([len(arg) for arg in args])
    mi =np.array([np.mean(arg) for arg in args])
    vi =np.array([np.var(arg,ddof=1) for arg in args])
    wi = ni/vi

    tmp =sum((1-wi/sum(wi))**2 / (ni-1))
    tmp /= (k**2 -1)

    dfbn = k - 1
    dfwn = 1 / (3 * tmp)

    m = sum(mi*wi) / sum(wi)
    f = sum(wi * (mi - m)**2) /((dfbn) * (1 + 2 * (dfbn - 1) * tmp))
    prob = scipy.special.fdtrc(dfbn, dfwn, f)   # equivalent to stats.f.sf
    return F_onewayResult(f, prob)

# Function to create the autocorrelation plots
def acf_plot(series):
    
    # Create subplots figure
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Partial Autocorrelation", 
                                                        "Autocorrelation"))

    # Partial Autocorrelation Plot (PACF)
    corr_array = pacf((series).dropna(), alpha=0.05) 
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]
    
    # Autocorrelation Plot (ACF)
    corr_array_acf = acf((series).dropna(), alpha=0.05) 
    lower_acf = corr_array_acf[1][:,0] - corr_array_acf[0]
    upper_acf = corr_array_acf[1][:,1] - corr_array_acf[0]

    
    # Create PACF plot
    [fig.add_trace(go.Scatter(x=(x,x), 
                              y=(0,corr_array[0][x]), 
                              mode='lines',
                              line_color='#3f3f3f',
                              name = 'PACF'),
                   row=1,col=1)

    # Loop through to plot each lag in the figure
    for x in range(len(corr_array[0]))]

    # Create trace for PACF
    fig.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), 
                             y=corr_array[0], 
                             mode='markers', 
                             marker_color='#090059',
                             marker_size=12,
                             name = 'PACF'),
                  row=1,col=1)
    
    # Create trace Upper Bound trace
    fig.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), 
                             y=upper_y, 
                             mode='lines', 
                             line_color='rgba(255,255,255,0)',
                             name = 'Upper Bound'),
                 row=1,col=1)
                  
    # Create trace Lower Bound trace
    fig.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), 
                             y=lower_y, 
                             mode='lines',
                             fillcolor='rgba(32, 146, 230,0.3)',
                             fill='tonexty', 
                             line_color='rgba(255,255,255,0)',
                             name = 'Lower Bound'),
                 row=1,col=1)

    
    

    # Create ACF figure 
    [fig.add_trace(go.Scatter(x=(x,x), 
                              y=(0,corr_array_acf[0][x]), 
                              mode='lines',
                              line_color='#3f3f3f',
                              name = 'ACF'), 
                   row=1,col=2)
    
    # Loop through to plot each lag in the figure
    for x in range(len(corr_array_acf[0]))]


    fig.add_trace(go.Scatter(x=np.arange(len(corr_array_acf[0])), 
                             y=corr_array_acf[0], 
                             mode='markers', 
                             marker_color='#090059',
                             marker_size=12,
                             name = 'ACF'),
                  row=1,col=2)

    fig.add_trace(go.Scatter(x=np.arange(len(corr_array_acf[0])), 
                             y=upper_acf, 
                             mode='lines', 
                             line_color='rgba(255,255,255,0)',
                             name = 'Upper Bound'),
                  row =1, col=2)

    fig.add_trace(go.Scatter(x=np.arange(len(corr_array_acf[0])),
                             y=lower_acf, 
                             mode='lines',
                             fillcolor='rgba(32, 146, 230,0.3)',
                             fill='tonexty', 
                             line_color='rgba(255,255,255,0)',
                             name = 'Lower Bound'),
                  row=1,col=2)

    
    
    
    # Update Figures
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1,15],showspikes=True)
    fig.update_yaxes(zerolinecolor='#000000',showspikes=True)
    
    #title='Partial Autocorrelation (PACF)' if plot_pacf else 'Autocorrelation (ACF)'
    fig.update_layout(template='none', 
                      title_font_color="#000000",
                      font_family='arial',
                      height = 300,
                      width = 710,
                      font_size=14,
                      margin=dict(l=60, r=60, t=50, b=50))
    fig.update_annotations(font=dict(family="Arial", size=16, color="black"))
        
    # Return the subplots
    return fig

def plotSeasonalDecompose(
    x,
    model='additive',
    filt=None,
    period=7,
    two_sided=True,
    extrapolate_trend=0,
    title="Seasonal Decomposition"):
    
    result = seasonal_decompose(
            x, model=model, filt=filt, period=7,
            two_sided=two_sided, extrapolate_trend=extrapolate_trend)
    fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"])
    fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.observed, mode='lines', name='Observed'),
                row=1, col=1,
            )

    fig.add_trace(
            go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name='Trend'),
                row=2, col=1,
            )

    fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name='Seasonal'),
                row=3, col=1,
            )

    fig.add_trace(
            go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name='Residuals'),
                row=4, col=1,
            )
    
    fig.update_layout(template = 'none',
                      title_font_color="#000000",
                      font_family='arial',
                      width = 710,
                      height = 500,
                      showlegend=False)
    
    fig.update_annotations(font=dict(family="Arial", size=16, color="black"))

    return fig

# MAPE
def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return round(np.mean(np.abs((y_true - y_pred) / y_true)),4) * 100


# CSS Style
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": "12rem",
    "width": "14rem",
    "padding": "2rem 1rem",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    'fontFamily': 'arial',
    'fontColor': 'white',
               
}



sidebar = html.Div(
    [
        html.H6("Menu", className="display-4", style={'fontFamily': 'arial',
                                                      'textAlign':'center',
                                                      'fontSize': 35,
                                                      'header_height':'6rem',
                                                      'footer_height':"12rem",
                                                      'color': '#000000'}),
        html.Hr(),
        html.P(
            "", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Overview", href="/", active="exact",style={'fontFamily': 'arial',
                                                                       'textAlign':'center',
                                                                       'color': '#000000'}),
                

                #html.Br(),               
                dbc.NavLink("Statistics", href="/page-1", active="exact",style={'fontFamily': 'arial',
                                                                       'textAlign':'center',
                                                                       'color': '#000000'}),


                #html.Br(),               
                dbc.NavLink("Visualisations", href="/page-2", active="exact",style={'fontFamily': 'arial',
                                                                       'textAlign':'center',
                                                                       'color': '#000000'}),
                
   
                #html.Br(),
                dbc.NavLink("Hierarquical Clustering", href="/page-3", active="exact",style={'fontFamily': 'arial',
                                                                       'textAlign':'center',
                                                                       'color': '#000000'}),

                #html.Br(),               
                dbc.NavLink("Seasonal Decomposition", href="/page-4", active="exact",style={'fontFamily': 'arial',
                                                                       'textAlign':'center',
                                                                       'color': '#000000'}),

                #html.Br(),
                dbc.NavLink("ARIMA", href="/page-5", active="exact",style={'fontFamily': 'arial',
                                                                       'textAlign':'center',
                                                                       'color': '#000000'}),
            ],
            vertical=True,
            pills=True,
            
        ),
        html.Hr(), # Horizontal Line
        
        # Add the Summary
        html.Div([html.H6("Summary", className="display-4", style={'fontFamily': 'arial',
                                                      'textAlign':'center',
                                                      'color': '#000000',
                                                      'fontSize': 20,
                                                      'header_height':'6rem',
                                                      'footer_height':"12rem"}),
                  html.P("The data was gathered from CSO's database. This dataset contains 100,800 records and 7 features. There is a high level of granularity in the data which allows cross-filtering by the Statistic, Gender and Age Group. Hierarquical Clustering Agglomerative and ARIMA models were tested on this data.",
                         style={'fontSize': 12,
                                'fontFamily': 'arial',
                                'fontFamily': 'verdana',
                                'color': '#000000',
                                'textAlign': 'left'}),
                              html.Hr(),
                  
                  # Data Source
                  html.Label(['Developer: ',
                              html.A('Giuliano Silva', 
                                     href='https://www.linkedin.com/in/giulianomsilva/',
                                     target="_blank",
                                     style={'fontSize': 14, 'color':'blue', 'fontFamily': 'arial'})]),
                  html.Label(['Data Source: ',
                              html.A('CSO', 
                                     href='https://www.cso.ie/en/databases/',
                                     target="_blank",
                                     style={'fontSize': 14, 'color':'blue', 'fontFamily': 'arial'})]),
                  
])        

    ],
    style=SIDEBAR_STYLE,
)


content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)


# APP layout
app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return [
                html.H1('Summary',style={'textAlign':'center',
                                         'fontFamily': 'arial',
                                         'fontSize': 35,
                                         'color': '#000000'}),
            html.Hr(),
            html.Div(
                className="row", children=[
                    html.Div(className='six columns', children=[
                        dcc.Dropdown(
                            id='stats-dropdown',
                            options=[{'label': i, 'value': i}
                                     for i in df['Statistic'].unique()],
                            placeholder="Select  Statistic",
                            style=dict(
                                width='100%',
                                verticalAlign="center",
                                justifyContent = 'center',
                                fontFamliy = 'Arial'),
                            value='In-Patients')], style={'width': '25%',
                                                          'font-family': 'arial'})
            # Age Dropdown
            , html.Div(className='six columns', children=[
        dcc.Dropdown(id='age-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Age Group'].unique()],
                     placeholder="Select  Age Group",
                     style=dict(
                         width='100%',
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Arial'),
                     value='25 - 34 years')],  style={'width': '25%',
                                                      'font-family': 'arial'})
            # Gender Dropdown
            , html.Div(className='six columns', children=[
        dcc.Dropdown(id='gender-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Sex'].unique()],
                     placeholder="Select  Gender",
                     style=dict(
                         width='100%',
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Arial'),
                     value='Female')],  style={'width': '25%',
                                               'font-family': 'arial',
                                                'display':'inline-block'}) 
            # Area Dropdown
            , html.Div(className='six columns', children=[
        dcc.Dropdown(id='county-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Area'].unique()],
                     placeholder="Select  Area",
                     style=dict(
                         width='100%',
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Arial'),
                     value='Dublin')], style={'width': '25%',
                                              'font-family': 'arial'})

                ], style=dict(display='flex')),
            


            html.Div([dcc.Graph(id='pie-graph',style={'display':'inline-block', 
                                                'width': '40%'}),
             
            dcc.Graph(id='treemap-graph',style={'display':'inline-block',
                                                'width': '60%'})],style=dict(display='flex')),
   
            dcc.Graph(id='ie-map',style={'display':'inline-block','width': '100%'}),
            
                ]
    
    

    
    elif pathname == "/page-1":
        return [
            dbc.Container(
            [
                html.H1("Statistical Analysis", style={'textAlign':'center',
                                                       'fontFamily': 'arial',
                                                       'fontSize': 35,
                                                       'color': '#000000'}),
                html.Hr(),
        
    html.Div(
        className="row", children=[
            html.Div(className='six columns', children=[
                dcc.Dropdown(
                    id='stats-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Statistic'].unique()],
                     placeholder="Select  Statistic",
                     style=dict(
                         width='100%',
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Arial'),
                     value='In-Patients')],  style={'width': '33.33%',
                                                    'font-family': 'arial'})  
            # Age Dropdown
            , html.Div(className='six columns', children=[
        dcc.Dropdown(id='age-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Age Group'].unique()],
                     placeholder="Select  Age Group",
                     style=dict(
                         width='100%',
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Arial'),
                     value='25 - 34 years')],  style={'width': '33.33%',
                                                      'font-family': 'arial'})

            # Area Dropdown
            , html.Div(className='six columns', children=[
        dcc.Dropdown(id='county-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Area'].unique()],
                     placeholder="Select  Area",
                     style=dict(
                         width='100%',
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Arial'),
                     value='Dublin')], style={'width': '33.33%',
                                              'font-family': 'arial'})
        ], style=dict(display='flex')),  
                
                html.Br(),
                
                html.Div([
                dcc.Graph(id='stats-table',style={'display':'inline-block','width': '50%'}),

                dcc.Graph(id='stats-table-2',style={'display':'inline-block','width': '50%'})],style=dict(display='flex')),
                
                html.Div([                
                dcc.Graph(id='qq_plot-graph', style={'display':'inline-block','width': '33.33%'}),
                
                dcc.Graph(id='distplot-graph',style={'display':'inline-block','width': '33.33%'}),

                dcc.Graph(id='qq_plot-graph-2', style={'display':'inline-block','width': '33.33%'})],style=dict(display='flex')),



            ],
            fluid=True,
        )

                ]
   
    
    elif pathname == "/page-2":
        return [
                html.H1('Visualisations',style={'textAlign':'center',
                                         'fontFamily': 'arial',
                                         'fontSize': 35,
                                         'color': '#000000'}),
                html.Hr(),
        
            # Stats Dropdown
           html.Div(
        className="row", children=[
            html.Div(className='six columns', children=[
                dcc.Dropdown(
                    id='stats-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Statistic'].unique()],
                     placeholder="Select  Statistic",
                     style=dict(
                         width='100%',
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Arial'),
                     value='In-Patients')], style={'width': '25%',
                                               'font-family': 'arial',
                                                'display':'inline-block'}), 
            
    
            # Age Dropdown
             html.Div(className='six columns', children=[
        dcc.Dropdown(id='age-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Age Group'].unique()],
                     placeholder="Select  Age Group",
                     style=dict(
                         width='100%',
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Arial'),
                     value='25 - 34 years')], style={'width': '25%',
                                               'font-family': 'arial',
                                                'display':'inline-block'})  
            # Age Dropdown
            , html.Div(className='six columns', children=[
        dcc.Dropdown(id='county-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Area'].unique()],
                     placeholder="Select  Area",
                     style=dict(
                         width='100%',
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Arial'),
                     value='Dublin')],  style={'width': '25%',
                                               'font-family': 'arial',
                                                'display':'inline-block'}) 
            # Age Dropdown
            , html.Div(className='six columns', children=[
        dcc.Dropdown(id='gender-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Sex'].unique()],
                     placeholder="Select  Gender",
                     style=dict(
                         width='100%',
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Arial'),
                     value='Female')],  style={'width': '25%',
                                               'font-family': 'arial',
                                                'display':'inline-block'}) 
            
                    ], style=dict(display='flex')),    
            
                html.Br(),
            
                html.Div(className='six columns', children=[
                dcc.Graph(id='gender-graph',style={'display':'inline-block','width': '50%'}),
                dcc.Graph(id='age-graph',style={'display':'inline-block','width': '50%'})]),
            
                html.Div(className='six columns', children=[
                dcc.Graph(id='bar_plot-graph',style={'display':'inline-block','width': '50%'}),
                dcc.Graph(id='county-graph',style={'display':'inline-block','width': '50%'})]),
            
                ]


    elif pathname == "/page-3":
        return [
        html.H1("Dendrogram",style={'textAlign':'center',
                                         'fontFamily': 'arial',
                                         'fontSize': 35,
                                         'color': '#000000'}),
        html.Hr(),

    html.Div(
        className="row", children=[
            html.Div(className='six columns', children=[
                dcc.Dropdown(
                    id='stats-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Statistic'].unique()],
                     placeholder="Select  Statistic",
                     style=dict(
                         width='100%',
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Arial'),
                     value='In-Patients')],  style={'width': '50%','font-family': 'arial'})  
            # Age Dropdown
            , html.Div(className='six columns', children=[
        dcc.Dropdown(id='age-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Age Group'].unique()],
                     placeholder="Select  Age Group",
                     style=dict(
                         width='100%',
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Arial'),
                     value='25 - 34 years')],  style={'width': '50%','font-family': 'arial'})  
        ], style=dict(display='flex')),  
            
            html.Br(),
            dcc.Graph(id='male-dendrogram-graph',style={'display':'inline-block','width': '50%'}),
            dcc.Graph(id='female-dendrogram-graph',style={'display':'inline-block','width': '50%'}),
                
        
        ]

    elif pathname == "/page-4":
        return [
        html.H1("Seasonal Decomposition",style={'textAlign':'center',
                                         'fontFamily': 'arial',
                                         'fontSize': 35,
                                         'color': '#000000'}),
        html.Hr(),

    html.Div(
        className="row", children=[
            html.Div(className='six columns', children=[
                dcc.Dropdown(
                    id='stats-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Statistic'].unique()],
                     placeholder="Select  Statistic",
                     style=dict(
                         width='100%',
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Arial'),
                     value='In-Patients')],  style={'width': '33.33%','font-family': 'arial'})  
            # Age Dropdown
            , html.Div(className='six columns', children=[
        dcc.Dropdown(id='age-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Age Group'].unique()],
                     placeholder="Select  Age Group",
                     style=dict(
                         width='100%',
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Arial'),
                     value='25 - 34 years')],  style={'width': '33.33%','font-family': 'arial'})  
            
            , html.Div(className='six columns', children=[
        dcc.Dropdown(id='county-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Area'].unique()],
                     placeholder="Select  Area",
                     style=dict(
                         width='100%',
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Arial'),
                     value='Dublin')],  style={'width': '33.33%','font-family': 'arial'})
        ], style=dict(display='flex')),  
            
            html.Br(),
            dcc.Graph(id='pacf-graph-1',style={'display':'inline-block','width': '50%'}),
            dcc.Graph(id='pacf-graph-2',style={'display':'inline-block','width': '50%'}),
            dcc.Graph(id='seasonal-1',style={'display':'inline-block','width': '50%'}),
            dcc.Graph(id='seasonal-2',style={'display':'inline-block','width': '50%'}),
                
        
        ]    
    
    
    
    elif pathname == "/page-5":
        return [
            dbc.Container(
            [
                html.H1("ARIMA model",style={'textAlign':'center',
                                         'fontFamily': 'arial',
                                         'fontSize': 35,
                                         'color': '#000000'}),
                html.Hr(),
        #html.Div(id="tab-content", className="p-4"),
        
    html.Div(
        className="row", children=[
            html.Div(className='six columns', children=[
                dcc.Dropdown(
                    id='stats-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Statistic'].unique()],
                     placeholder="Select  Statistic",
                     style=dict(
                         width='100%',
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Arial'),
                     value='In-Patients')],  style={'width': '25%','font-family': 'arial'})
            # Age Dropdown
            , html.Div(className='six columns', children=[
        dcc.Dropdown(id='age-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Age Group'].unique()],
                     placeholder="Select  Age Group",
                     style=dict(
                         width='100%',
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Arial'),
                     value='25 - 34 years')],  style={'width': '25%','font-family': 'arial'})
            # Age Dropdown
            , html.Div(className='six columns', children=[
        dcc.Dropdown(id='county-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Area'].unique()],
                     placeholder="Select  Area",
                     style=dict(
                         width='100%',
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Arial'),
                     value='Dublin')],  style={'width': '25%','font-family': 'arial'})

            # Area Dropdown
            , html.Div(className='six columns', children=[
        dcc.Dropdown(id='trend-dropdown',
                     options=[{'label': 'Constant', 'value': 'c'},
                              {'label': 'Trend', 'value': 't'},
                              {'label': 'Constant + Time', 'value': 'ct'},
                              {'label': 'Non-trend', 'value': 'n'}],
                     placeholder="Select  Trend",
                     style=dict(
                         width='100%',
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Arial'),
                     value='n')], style={'width': '25%','font-family': 'arial'}),  
        ], style=dict(display='flex')),  
                
                # Sliders (p,o,q, power) 1
                html.Div([
                    html.Div([
                    daq.Slider(
                        id='3DHR Slider',
                        min=0,
                        max=20,
                        value=2,
                        #marks={i: f"p {i}" for i in range(0,20, 1)},
                        handleLabel={"showCurrentValue": True,"label": "p"},
                        color = 'black',
                        step=1,
                        size=100)],
                        style={'display':'inline-block',
                               'margin-left':'0px',
                               'margin-top':'50px','width': '16.66'}),
                    html.Div([
                    daq.Slider(
                        id='3DHR Slider2',
                        min=0,
                        max=20,
                        value=0,
                        #marks={i: f"d {i}" for i in range(0,20, 1)},
                        handleLabel={"showCurrentValue": True,"label": "q"},
                        color = 'black',
                        step=1,
                        size=100)],
                        style={'display':'inline-block',
                               'margin-left':'20px',
                               'margin-top':'50px','width': '16.66'}),

                    html.Div([
                        daq.Slider(
                                id='3DHR Slider3',
                                min=0,
                                max=20,
                                value=0,
                                #marks={i: f"d {i}" for i in range(0,20, 1)},
                                handleLabel={"showCurrentValue": True,"label": "d"},
                                color = 'black',
                                step=1,
                        size=100)],
                        style={'display':'inline-block',
                               'margin-left':'20px',
                               'margin-top':'50px','width': '16.66'}),
                    html.Div([
                    daq.Slider(
                        id='3DHR Slider4',
                        min=0,
                        max=20,
                        value=2,
                        #marks={i: f"p {i}" for i in range(0,20, 1)},
                        handleLabel={"showCurrentValue": True,"label": "p"},
                        color = 'black',
                        step=1,
                        size=100)],
                        style={'display':'inline-block',
                               'margin-left':'380px',
                               'margin-top':'50px','width': '16.66'}),
                    html.Div([
                    daq.Slider(
                        id='3DHR Slider5',
                        min=0,
                        max=20,
                        value=0,
                        #marks={i: f"d {i}" for i in range(0,20, 1)},
                        handleLabel={"showCurrentValue": True,"label": "q"},
                        color = 'black',
                        step=1,
                        size=100)],
                        style={'display':'inline-block',
                               'margin-left':'20px',
                               'margin-top':'50px','width': '16.66'}),

                    html.Div([
                        daq.Slider(
                                id='3DHR Slider6',
                                min=0,
                                max=20,
                                value=0,
                                #marks={i: f"d {i}" for i in range(0,20, 1)},
                                handleLabel={"showCurrentValue": True,"label": "d"},
                                color = 'black',
                                step=1,
                        size=100)],
                        style={'display':'inline-block',
                               'margin-left':'20px',
                               'margin-top':'50px','width': '16.66'}),

                ], style={'margin-top': '20px'}),
                

                
            html.Br(),
            html.Div(dcc.Graph(id='bic-male'), style={'display':'inline-block','width': '50%'}),
            html.Div(dcc.Graph(id='bic-female'), style={'display':'inline-block','width': '50%'}),
            html.Div(dcc.Graph(id='ts-graph-male'), style={'display':'inline-block','width': '50%'}),
            html.Div(dcc.Graph(id='ts-graph-female'), style={'display':'inline-block','width': '50%'}),

            ],
            fluid=True,
        )

                ]    

# If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


# Callbacks & Functions for Summary


@app.callback(
    Output(component_id='pie-graph', component_property='figure'),  
    Input(component_id='stats-dropdown', component_property='value'),
    Input(component_id='age-dropdown', component_property='value'),
    Input(component_id='county-dropdown', component_property='value'),

)

def pie_chart(selected_stat, selected_age, selected_area):
    colorscale = ['#1E90FF','#997070']
    colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
    df_2 = df[(df['Statistic']==selected_stat) &
              (df['Age Group']==selected_age) &
              (df['Area']==selected_area)]
    df_2 = df_2[[ 'Sex', 'VALUE', 'Statistic', 'Area', 'Age Group']].groupby(['Sex'], as_index=False).sum().sort_values('VALUE', ascending=True)
    df_2 = df_2[['Sex', 'VALUE']]
    fig = go.Figure(data=[go.Pie(labels=['Male', 'Female'],
                                 textinfo='label',
                                 values=df_2.VALUE)])
    fig.update_traces(hoverinfo='label+percent', 
                      textinfo='percent+label', 
                      textfont_size=14,
                      marker=dict(colors=colorscale, 
                                  line=dict(color='#000000', width=1)))
    
    
    fig.update_layout(margin = dict(t=50, l=80, r=25, b=25),
                      showlegend=False,
                      title_font_color="#000000",
                      height=450, 
                      width=400)
    return fig

@app.callback(
    Output(component_id='treemap-graph', component_property='figure'),
    Input(component_id='stats-dropdown', component_property='value'),

    Input(component_id='county-dropdown', component_property='value'),
)

def treemap(selected_stat, selected_area):
    df_3 = df[(df['Statistic']==selected_stat) &
              (df['Area']==selected_area)]
    fig = px.treemap(df_3, 
                     path=['Statistic',  'Sex', 'Age Group'], 
                     values='VALUE',
                     color = 'Sex',
                     color_discrete_map={'(?)':'white','Female': '#997070','Male':'#1E90FF'})
    fig.update_layout(margin = dict(t=50, l=25, r=0, b=25),
                      title_font_color="#000000",
                      height=450, 
                      width=870)
    return fig



@app.callback(
    Output(component_id='ie-map', component_property='figure'),  
    Input(component_id='stats-dropdown', component_property='value'),
    Input(component_id='age-dropdown', component_property='value'),
    Input(component_id='gender-dropdown', component_property='value')
)

def ie_map(selected_stat, selected_age, selected_gender):
    api_token = "pk.eyJ1IjoiZ2l1c2lsdmE4OSIsImEiOiJja3h6dXRrYjczNGJ6MnFtdndvcWkxbWNpIn0.hmvr8fms4u5K_1aGxoM9AA"
    
    df_1 = df[(df['Statistic']==selected_stat) &
              (df['Age Group']==selected_age) &
              (df['Sex']==selected_gender)]
    # Group by the categorical variable and calculatute the mean of the numerical
    df_1 = df_1[['Area',  'VALUE', 'Statistic', 'Age Group', 'Sex']]\
    .groupby(['Area', 'Age Group', 'Sex'], as_index = False).mean()\
    .sort_values('VALUE', ascending=True)
    
    # Filter out non-County variables
    df_1 = df_1[(df_1['Area'] != 'Ireland excl. non-residents') &
                (df_1['Area'] !='Non-residents')]
    
    

    fig= go.Figure(go.Choroplethmapbox(z=df_1.VALUE, # This is the data.
                                       locations=df_1.Area,
                                       colorscale='blues',
                                       colorbar=dict(thickness=20, ticklen=3),
                                       geojson=jdata,
                                       text=locations,
                                       hoverinfo='all',
                                       marker_line_width=1, 
                                       marker_opacity=0.75))


    fig.update_layout(mapbox = dict(center= dict(lat=53.425049,  lon=-7.944620),
                                     accesstoken= api_token,
                                     style='basic',
                                     zoom=5.6,
                                   ));

    # Use update_layout in order to define few configuration such as figure height and width, title, etc
    fig.update_layout(
        title='',
        height=450, # Figure height
        width=1470, # Figure width
        font_family="Arial", # Set Font style
        font_size=15,
        template='none',
        title_font_color="#000000",
        margin = dict(t=50, l=0, r=0, b=0))

    # Display figure
    return fig






# Callback & Function for Stats
@app.callback(
    Output(component_id='stats-table', component_property='figure'),
    [
        Input(component_id='stats-dropdown', component_property='value'),
        Input(component_id='county-dropdown', component_property='value'),
        Input(component_id='age-dropdown', component_property='value')

    ],
)


def stats_table(selected_stat, selected_area, selected_age):
    # Data Prep
    
    df_1 = df[df['Statistic']==selected_stat]
    df_2 = df_1[['Statistic', 'Area', 'VALUE', 'Sex', 'Age Group', 'Year']]\
    .groupby(['Statistic', 'Area', 'Sex', 'Age Group', 'Year'], as_index=False)\
    .mean().sort_values('Area', ascending=True)

    df_2 = df_2[df_2['Area']==selected_area]
    df_2 = df_2[df_2["Age Group"]==selected_age]
    df_2 = df_2[['VALUE', 'Area', 'Sex', 'Age Group', 'Year']]
    df_male_1 = df_2.query("Sex=='Male'")
    df_female_1 = df_2.query("Sex=='Female'")

    # Style
    colorscale = [[0, '#1434A4'],[.1, '#ffffff'],[1, '#ffffff']]

    # Shapiro test for nomarlity
    stat, p = shapiro(df_male_1['VALUE'])

    # Shapiro test for nomarlity
    stat_2, p_2 = shapiro(df_female_1['VALUE'])

    # Levene test for Equality Variances
    stat_3, p_3 = stats.levene(df_male_1['VALUE'], df_female_1['VALUE'])

    #One-way ANOVA test
    stat_4, p_4  = welch_anova_np(df_male_1['VALUE'], df_female_1['VALUE'], var_equal = False)


    # Round Stats and p-values for Shapiro test (Male)
    p = round(p, 4)
    stat = round(stat, 4)

    # Round Stats and p-values for Shapiro test (Female)
    p_2 = round(p_2, 4)
    stat_2 = round(stat_2, 4)                        


     # Round Stats and p-values for Levene test (Male vs Female)
    p_3 = round(p_3, 4)
    stat_3 = round(stat_3, 4)     


    #Round stat and p-value
    p_4 = round(p_4, 4)
    stat_4 = round(stat_4, 4)


    # Define confidence interval
    alpha = 0.05


    # Condition 
    if p > alpha:
        msg = 'Accept'
    else:
        msg = 'Reject'


    # Condition 
    if p_2 > alpha:
        msg_2 = 'Accept'
    else:
        msg_2 = 'Reject'

    # Condition 
    if p_3 > alpha:
        msg_3 = 'Accept'
    else:
        msg_3 = 'Reject'

    # Condition for Anova
    if p_4 > 0.05:
        msg_4 = 'Accept'
    else:
        msg_4 = 'Reject'


    # Append results in a list
    result_mat = [
        ['Test','Length', 'T-Statistic', 'p-value', 'Comments'],
        ['Shapiro (M)',len(df_male_1), stat, p, msg],
        ['Shapiro (F)',len(df_male_1), stat_2, p_2, msg_2],
        ['Levene (M&F)',len(df_male_1), stat_3, p_3, msg_3],
        ['ANOVA (M&F)',len(df_male_1), stat_4, p_4, msg_4],

    ]


    # Create table
    stats_table = ff.create_table(result_mat,
                                  height_constant=20,
                                  colorscale=colorscale)

    # Add style to the table
    stats_table['data'][0]
    stats_table['layout']['height']=300
    stats_table['layout']['margin']['t']=70
    stats_table['layout']['margin']['b']=70
    stats_table['layout']['margin']['r']=20

    stats_table.update_layout(width=660)
    stats_table.update_layout(title_text="Shapiro, Levene & ANOVA Test",
                              title_font_color="#000000")

    # Show table
    return stats_table

# Callback & Function for Stats
@app.callback(
    Output(component_id='stats-table-2', component_property='figure'),
    [
        Input(component_id='stats-dropdown', component_property='value'),
        Input(component_id='county-dropdown', component_property='value'),
        Input(component_id='age-dropdown', component_property='value')

    ],
)

def stats_table_2(selected_stat, selected_area, selected_age):
    # Data Prep
    df_1 = df[df['Statistic']==selected_stat]
    df_2 = df_1[['Statistic', 'Area', 'VALUE', 'Sex', 'Age Group', 'Year']]\
    .groupby(['Statistic', 'Area', 'Sex', 'Age Group', 'Year'], as_index=False)\
    .mean().sort_values('Area', ascending=True)


    df_2 = df_2[df_2['Area']==selected_area]
    df_2 = df_2[df_2["Age Group"]==selected_age]
    df_2 = df_2[['VALUE', 'Area', 'Sex', 'Age Group', 'Year']]
    df_male_1 = df_2.query("Sex=='Male'")
    df_female_1 = df_2.query("Sex=='Female'")

    colorscale = [[0, '#1434A4'],[.1, '#ffffff'],[1, '#ffffff']]
    # Apply Chi-square test of independence of variables in a contingency table.
    df2_mk_male = df_male_1['VALUE'] 
    df2_mk_female = df_female_1['VALUE'] 
    trend, h, p, z, t, s, var_s, slope, intercept = mk.original_test(df2_mk_male)
    
    trend_2, h_2, p_2, z_2, t_2, s_2, var_s_2, slope_2, intercept_2 = mk.original_test(df2_mk_female)

    # Round Statistic and p-values
    z = round(z, 4)
    p = round(p, 4)
    z_2 = round(z_2,4)
    p_2 = round(p_2,4)
    
    # Define confidence interval
    alpha = 0.05
    
    # Condition 
    if p > alpha:
        msg = 'Accept'
    else:
        msg = 'Reject'
        
    # Condition 
    if p_2 > alpha:
        msg_2 = 'Accept'
    else:
        msg_2 = 'Reject'
        

    # Append results in a list
    result_mat = [
        ['Gender','Length', 'Trend', 'Direction', 'T-Statistic', 'p-value', 'Comments'],

        ["Male", len(df2_mk_male), h, trend, z, p, msg],
        ["Female", len(df2_mk_female), h_2, trend_2, z_2, p_2, msg_2]
   
    
    ]


    # Create table
    mann_kendall_table = ff.create_table(result_mat,
                                height_constant=20,
                                colorscale=colorscale)
    
    
        # Add style to the table
    mann_kendall_table['data'][0]
    mann_kendall_table['layout']['height']=300
    mann_kendall_table['layout']['margin']['t']=70
    mann_kendall_table['layout']['margin']['b']=70
    mann_kendall_table['layout']['margin']['r']=20
    
    mann_kendall_table.update_layout(width=700)
    mann_kendall_table.update_layout(title_text="Mann Kendall Test",
                                     title_font_color="#000000")


    # Show table
    return mann_kendall_table


# Callback & Function for Stats
@app.callback(
    Output(component_id='distplot-graph', component_property='figure'),
    [
        Input(component_id='stats-dropdown', component_property='value'),
        Input(component_id='county-dropdown', component_property='value'),
        Input(component_id='age-dropdown', component_property='value')

    ],
)


def distplot(selected_stat, selected_area, selected_age):
    
    df_1 = df[df['Statistic']==selected_stat]
    df_2 = df_1[['Statistic', 'Area', 'VALUE', 'Sex', 'Age Group', 'Year']]\
    .groupby(['Statistic', 'Area', 'Sex', 'Age Group', 'Year'], as_index=False)\
    .mean().sort_values('Area', ascending=True)

    df_2 = df_2[df_2['Area']==selected_area]
    df_2 = df_2[df_2["Age Group"]==selected_age]
    df_2 = df_2[['VALUE', 'Area', 'Sex', 'Age Group', 'Year']]
    df_male_1 = df_2.query("Sex=='Male'")
    df_female_1 = df_2.query("Sex=='Female'")

    hist_data = [df_male_1.VALUE, df_female_1.VALUE]

    group_labels = ['Male', 'Female']
    colors = ['#1E90FF','#997070']  

    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)
    
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="top",
    y = 1.02,
    xanchor="right",
    x=1))

    # Add title
    fig.update_layout(template='none',
                      title_text = 'Distribution Plot',
                      title_font_color="#000000",
                      width=400,
                      height=400,
                      margin=dict(l=50, r=0, t=50, b=50))
    return fig




# Callback & Function for Stats
@app.callback(
    Output(component_id='qq_plot-graph', component_property='figure'),
    [
        Input(component_id='stats-dropdown', component_property='value'),
        Input(component_id='county-dropdown', component_property='value'),
        Input(component_id='age-dropdown', component_property='value')

    ],
)



def qq_plot(selected_stat, selected_area, selected_age):
    
    df_1 = df[df['Statistic']==selected_stat]
    df_2 = df_1[['Statistic', 'Area', 'VALUE', 'Sex', 'Age Group', 'Year']]\
    .groupby(['Statistic', 'Area', 'Sex', 'Age Group', 'Year'], as_index=False)\
    .mean().sort_values('Area', ascending=True) 

    df_2 = df_2[df_2['Area']==selected_area]
    df_2 = df_2[df_2["Age Group"]==selected_age]
    df_2 = df_2[['VALUE', 'Area', 'Sex', 'Age Group', 'Year']]
    df_male_1 = df_2.query("Sex=='Male'")

    
    
    qqplot_data1 = qqplot(df_male_1['VALUE'], line='s').gca().lines    
    
    fig = go.Figure()

    # Ireland
    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data1[0].get_xdata(),
        'y': qqplot_data1[0].get_ydata(),
        'name': 'Male',
        'mode': 'markers',
        'marker': {
            'color': '#1434A4'
        }
    })

    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data1[1].get_xdata(),
        'y': qqplot_data1[1].get_ydata(),
        'name': 'QQ-Line',
        'mode': 'lines',
        'line': {
            'color': '#FF0000'}
    })


    fig['layout'].update({
        'title': 'Male Quantile-Quantile Plot',
        'xaxis': {
            'title': '',
            'zeroline': False
        },
        'yaxis': {
            'title': ''
        },
        'showlegend': False,
        'width': 450,
        'height': 400,
    })

    # Update figure
    fig.update_layout(plot_bgcolor='white',
                      title_font_color="#000000",
                      template = 'none',
                      margin=dict(l=40, r=0, t=50, b=50))

    return fig



# Callback & Function for Stats
@app.callback(
    Output(component_id='qq_plot-graph-2', component_property='figure'),
    [
        Input(component_id='stats-dropdown', component_property='value'),
        Input(component_id='county-dropdown', component_property='value'),
        Input(component_id='age-dropdown', component_property='value')

    ],
)



def qq_plot_2(selected_stat, selected_area, selected_age):
    
    df_1 = df[df['Statistic']==selected_stat]
    df_2 = df_1[['Statistic', 'Area', 'VALUE', 'Sex', 'Age Group', 'Year']]\
    .groupby(['Statistic', 'Area', 'Sex', 'Age Group', 'Year'], as_index=False)\
    .mean().sort_values('Area', ascending=True) 

    df_2 = df_2[df_2['Area']==selected_area]
    df_2 = df_2[df_2["Age Group"]==selected_age]
    df_2 = df_2[['VALUE', 'Area', 'Sex', 'Age Group', 'Year']]
    df_female_1 = df_2.query("Sex=='Female'")

    
    
    qqplot_data1 = qqplot(df_female_1['VALUE'], line='s').gca().lines    
    
    fig = go.Figure()

    # Ireland
    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data1[0].get_xdata(),
        'y': qqplot_data1[0].get_ydata(),
        'name': 'Female',
        'mode': 'markers',
        'marker': {
            'color': '#1434A4'
        }
    })

    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data1[1].get_xdata(),
        'y': qqplot_data1[1].get_ydata(),
        'name': 'QQ-Line',
        'mode': 'lines',
        'line': {
            'color': '#FF0000'}
    })


    fig['layout'].update({
        'title': 'Female Quantile-Quantile Plot',
        'xaxis': {
            'title': '',
            'zeroline': False
        },
        'yaxis': {
            'title': ''
        },
        'showlegend': False,
        'width': 450,
        'height': 400,
    })

    # Update figure
    fig.update_layout(plot_bgcolor='white',
                      title_font_color="#000000",
                      template = 'none',
                      margin=dict(l=70, r=0, t=50, b=50))

    return fig


# Callbacks & Function for Visualisations pt 1
@app.callback(
    Output(component_id='gender-graph', component_property='figure'),
    [
        Input(component_id='stats-dropdown', component_property='value'),
        Input(component_id='age-dropdown', component_property='value'),
        Input(component_id='county-dropdown', component_property='value')
    ],
)

def scatter(selected_stat, selected_age, selected_area):
    
    df_2 = df[['Statistic','VALUE', 'Year', 'Sex', 'Age Group', 'Area']]

    df_2 = df_2[(df_2['Statistic']==selected_stat) &
                (df_2['Age Group']==selected_age) &
                (df_2['Area']==selected_area)]

    df_2 = df_2.groupby(['Year', 'Statistic', 'Sex', 'Age Group', 'Area'], as_index=False)\
    .sum().sort_values('Year', ascending=True)


    df_male = df_2.query("Sex=='Male'")
    df_female = df_2.query("Sex=='Female'")


    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=df_male.Year, y=df_male.VALUE,
                        mode='lines',
                        line=dict(color='#1E90FF', width=2),
                        name='Male'))
    fig.add_trace(go.Scatter(x=df_female.Year, y=df_female.VALUE,
                        mode='lines+markers',
                        line=dict(color='#997070', width=2),
                        name='Female'))

    fig.update_layout(template='none', 
                      width=700, 
                      height = 450,
                      margin = dict(t=50, l=30, r=0, b=50))
    
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y = 1.02,
    xanchor="left",
    x=0.02))

    return fig


# Callbacks & Function for Visualisations pt 1
@app.callback(
    Output(component_id='age-graph', component_property='figure'),
    [
        Input(component_id='stats-dropdown', component_property='value'),
        Input(component_id='county-dropdown', component_property='value'),
        Input(component_id='gender-dropdown', component_property='value'),
    ],
)

def scatter(selected_stat, selected_area, selected_gender):
    
    df_2 = df[['Statistic','VALUE', 'Year', 'Sex', 'Age Group', 'Area']]

    df_2 = df_2[(df_2['Statistic']==selected_stat) &
                (df_2['Area']==selected_area) &
                (df_2['Sex']==selected_gender)]

    df_2 = df_2.groupby(['Year', 'Statistic', 'Age Group', 'Area'], as_index=False)\
    .sum().sort_values('Year', ascending=True)

    fig = px.line(df_2, x="Year", 
                  y="VALUE", 
                  color="Age Group",
                  color_discrete_sequence=px.colors.qualitative.Safe)
    
    fig.update_xaxes(title_text = "")
    fig.update_yaxes(title_text = "")

    fig.update_layout(template='none', 
                      width=700, 
                      height = 450,
                      margin = dict(t=50, l=80, r=0, b=50))
    
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1))

    return fig

@app.callback(
    Output(component_id='bar_plot-graph', component_property='figure'),
    Input(component_id='stats-dropdown', component_property='value')
)


def bar_plot(selected_stat):

    df_1 = df[['Statistic', 'Age Group', 'Sex', 'VALUE']]\
    .groupby(['Statistic', 'Age Group', 'Sex'], as_index=False)\
    .mean().sort_values('Age Group', ascending=True)
    df_1 = df_1[df_1['Statistic']==selected_stat]
    df_1 = df_1[['Age Group', 'Sex', 'VALUE']]
    df_male_1 = df_1.query("Sex=='Male'")
    df_female_1 = df_1.query("Sex=='Female'")

    fig = go.Figure()
   
    fig.add_trace(go.Histogram(x=df_male_1['Age Group'], 
                               y=df_male_1['VALUE'],
                               histfunc="sum", 
                               name = 'Male',
                               marker={'color': '#1E90FF'}))
    
    fig.add_trace(go.Histogram(x=df_female_1['Age Group'], 
                               y=df_female_1['VALUE'],
                               histfunc="sum", 
                               name='Female',
                               marker={'color': '#997070'}))
    
    fig.update_traces(marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, 
                      opacity=0.6)
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="top",
    y = 1.02,
    xanchor="right",
    x=1))

    fig.update_layout(barmode='group',
                      template='none',
                      margin = dict(t=50, l=20, r=0, b=200),
                      height=550,
                      width=700)
    
    return fig

# Callbacks & Function for Visualisations pt 2
@app.callback(
    Output(component_id='county-graph', component_property='figure'),
    [
        Input(component_id='stats-dropdown', component_property='value'),
        Input(component_id='age-dropdown', component_property='value')
    ],
)


def bar_plot_2(selected_stat, selected_age):
    
    
    df_2 = df[(df['Statistic']==selected_stat) &
                (df['Age Group']==selected_age)]
    
    df_2 = df_2[['Statistic', 'Area', 'VALUE', 'Sex', 'Age Group']]\
    .groupby(['Statistic', 'Area', 'Sex', 'Age Group'], as_index=False)\
    .mean().sort_values('VALUE', ascending=False)
    df_2 = df_2[['VALUE', 'Area', 'Sex', 'Age Group']]
    df_male_1 = df_2.query("Sex=='Male'")
    df_female_1 = df_2.query("Sex=='Female'")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_male_1['Area'], 
                               y=df_male_1['VALUE'],
                               histfunc="sum", 
                               name='Male',
                               marker={'color': '#1E90FF'}))
    
    fig.add_trace(go.Histogram(x=df_female_1['Area'], 
                               y=df_female_1['VALUE'],
                               histfunc="sum", 
                               name='Female',
                               marker={'color': '#997070'}))                               
    
        
    fig.update_traces(marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, 
                      opacity=0.6)

    fig.update_layout(template='none',
                      title_font_color="#000000",
                      width=700,
                      height = 550,
                      margin = dict(t=50, l=50, r=0, b=200))
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="top",
    y = 1.02,
    xanchor="right",
    x=1))
    return fig




@app.callback(
    Output(component_id='male-dendrogram-graph', component_property='figure'),
    [
        Input(component_id='stats-dropdown', component_property='value'),
        Input(component_id='age-dropdown', component_property='value')

    ],
)


def dendrogram(selected_stat,selected_age):
    df_1 = df[df['Statistic']==selected_stat]

    df_2 = df_1[['Statistic', 'Area', 'VALUE', 'Sex','Age Group']]\
    .groupby(['Statistic', 'Area', 'Age Group','Sex'], as_index=False)\
    .mean().sort_values('Area', ascending=True) 

    df_2 = df_2[df_2["Age Group"]==selected_age]
    df_2 = df_2.query("Sex=='Male'")
    names = df_2.Area.unique()
    
    df_3 = df_2[['VALUE']]

    fig = ff.create_dendrogram(df_3, labels=names)
    fig.update_layout(width=710, 
                      height=600, 
                      title="Male",
                      title_font_color="#000000",
                      template = 'none',
                      margin=dict(l=40, r=20, t=50, b=200))
    return fig

@app.callback(
    Output(component_id='female-dendrogram-graph', component_property='figure'),
    [
        Input(component_id='stats-dropdown', component_property='value'),
        Input(component_id='age-dropdown', component_property='value')

    ],
)


def dendrogram_2(selected_stat,selected_age):
    df_1 = df[df['Statistic']==selected_stat]

    df_2 = df_1[['Statistic', 'Area', 'VALUE', 'Sex','Age Group']]\
    .groupby(['Statistic', 'Area', 'Age Group','Sex'], as_index=False)\
    .mean().sort_values('Area', ascending=True) 

    df_2 = df_2[df_2["Age Group"]==selected_age]
    df_2 = df_2.query("Sex=='Female'")
    names = df_2.Area.unique()
    
    df_3 = df_2[['VALUE']]

    fig = ff.create_dendrogram(df_3, labels=names)
    fig.update_layout(width=720, 
                      height=600, 
                      template = 'none',
                      title="Female",
                      title_font_color="#000000",
                      margin=dict(l=80, r=20, t=50, b=200))
    return fig

@app.callback(
    Output(component_id='pacf-graph-1', component_property='figure'),
    [
        Input(component_id='stats-dropdown', component_property='value'),
        Input(component_id='county-dropdown', component_property='value'),
        Input(component_id='age-dropdown', component_property='value')]
)

def auto_corr(selected_stat, selected_area, selected_age):

    df_2 = df[['Statistic','VALUE', 'Year', 'Sex', 'Age Group', 'Area']]

    df_2 = df_2[(df_2['Statistic']==selected_stat) &
                (df_2['Area']==selected_area) &
                (df_2['Age Group']==selected_age)]
    
    df_2 = df_2.query("Sex=='Male'")

    df_2 = df_2.groupby(['Year', 'Statistic', 'Age Group', 'Area'], as_index=False)\
    .mean().sort_values('Year', ascending=True)

    df_2 = df_2[['Year', 'VALUE']].set_index('Year')
    return acf_plot(df_2.VALUE)

@app.callback(
    Output(component_id='pacf-graph-2', component_property='figure'),
    [
        Input(component_id='stats-dropdown', component_property='value'),
        Input(component_id='county-dropdown', component_property='value'),
        Input(component_id='age-dropdown', component_property='value')]
)

def auto_corr_2(selected_stat, selected_area, selected_age):

    df_2 = df[['Statistic','VALUE', 'Year', 'Sex', 'Age Group', 'Area']]

    df_2 = df_2[(df_2['Statistic']==selected_stat) &
                (df_2['Area']==selected_area) &
                (df_2['Age Group']==selected_age)]
    
    df_2 = df_2.query("Sex=='Female'")

    df_2 = df_2.groupby(['Year', 'Statistic', 'Age Group', 'Area'], as_index=False)\
    .mean().sort_values('Year', ascending=True)
    
    df_2 = df_2[['Year', 'VALUE']].set_index('Year')
    return acf_plot(df_2.VALUE)




@app.callback(
    Output(component_id='seasonal-1', component_property='figure'),
    [
        Input(component_id='stats-dropdown', component_property='value'),
        Input(component_id='county-dropdown', component_property='value'),
        Input(component_id='age-dropdown', component_property='value')]
)

def seasonal_decompositon(selected_stat, selected_area, selected_age):

    df_2 = df[['Statistic','VALUE', 'Year', 'Sex', 'Age Group', 'Area']]

    df_2 = df_2[(df_2['Statistic']==selected_stat) &
                (df_2['Area']==selected_area) &
                (df_2['Age Group']==selected_age)]
    
    df_2 = df_2.query("Sex=='Male'")

    df_2 = df_2.groupby(['Year', 'Statistic', 'Age Group', 'Area'], as_index=False)\
    .mean().sort_values('Year', ascending=True).set_index('Year')


    return plotSeasonalDecompose(df_2.VALUE)


@app.callback(
    Output(component_id='seasonal-2', component_property='figure'),
    [
        Input(component_id='stats-dropdown', component_property='value'),
        Input(component_id='county-dropdown', component_property='value'),
        Input(component_id='age-dropdown', component_property='value')]
)

def seasonal_decompositon_2(selected_stat, selected_area, selected_age):

    df_2 = df[['Statistic','VALUE', 'Year', 'Sex', 'Age Group', 'Area']]

    df_2 = df_2[(df_2['Statistic']==selected_stat) &
                (df_2['Area']==selected_area) &
                (df_2['Age Group']==selected_age)]
    
    df_2 = df_2.query("Sex=='Female'")

    df_2 = df_2.groupby(['Year', 'Statistic', 'Age Group', 'Area'], as_index=False)\
    .mean().sort_values('Year', ascending=True)
    
    df_3 = df_2['VALUE']

    return plotSeasonalDecompose(df_3)




# Callback & Function for Stats
@app.callback(
    Output(component_id='bic-male', component_property='figure'),
    [
        Input(component_id='stats-dropdown', component_property='value'),
        Input(component_id='county-dropdown', component_property='value'),
        Input(component_id='age-dropdown', component_property='value'),
        Input(component_id='3DHR Slider', component_property='value'),
        Input(component_id='3DHR Slider2', component_property='value'),
        Input(component_id='3DHR Slider3', component_property='value'),
        Input(component_id='trend-dropdown', component_property='value'),        

    ],
)
def bic_mape(selected_stat, selected_area, selected_age,p,d,q, trend):
    
    # Apply period index with year frequency to the Year feature 
    df['Year'] = pd.PeriodIndex(df['Year'], freq='Y').to_timestamp()
    df_ts = df[df['Statistic']==selected_stat]
    df_test = df_ts[df_ts['Area']==selected_area]
    df_test = df_test[df_test['Age Group']==selected_age]
    df_test = df_test.query("Sex=='Male'")

    df_test = df_test[['Year', 'VALUE']].groupby('Year', as_index=False).mean().set_index('Year')

    # Create a variable to get the forecast off the diagnostic results
    mod = sm.tsa.statespace.SARIMAX(df_test, 
                                    order=(p,d,q),
                                    trend = trend,
                                    enforce_stationarity=True,
                                    enforce_invertibility=False)

    results = mod.fit(max_iter=100, method='powell')

    bic = results.bic
    
    # Create a variable basen on the best model and start to predict from 1986
    pred = results.get_prediction(start = pd.to_datetime('2006-01-01'), 
                                       dynamic = False)


    predictions = pd.DataFrame(pred.predicted_mean)

    # Create a variable to get the forecast off the diagnostic results
    pred_uc = results.get_forecast(steps=5, dynamic=True)

    # Create the confidence interval variable
    pred_ci = pred_uc.conf_int(0.05) # Set confidence interval of 5%

    # Confidence interval equals to the pred_ci variable calculate in the built-in function provided by pdmarima
    ci = pred_ci

    # Forecast the future
    future = pred_uc.predicted_mean
    
    mape_male = MAPE(df_test, predictions.predicted_mean)
    
    fig_trace_01 = go.Indicator(
    mode = "number",
    #gauge = {'shape': "bullet"},
    value = bic,
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    title = {'text': "BIC"})
    
    fig_trace_02 = go.Indicator(
    mode = "number",
    #gauge = {'shape': "bullet"},
    value = mape_male,
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    title = {'text': "MAPE (%)"})
    
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{'type' : 'indicator'}, 
                {'type' : 'indicator'}]])
    
    fig.append_trace(fig_trace_01, row=1, col=1)
    fig.append_trace(fig_trace_02, row=1, col=2)
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0),
                      height=80,
                      title_font_color="#000000",
                      font_family="Arial", # Set Font style
                      font_size=12) # Set Font size) # legend false 

    return fig



# Callback & Function for Stats
@app.callback(
    Output(component_id='bic-female', component_property='figure'),
    [
        Input(component_id='stats-dropdown', component_property='value'),
        Input(component_id='county-dropdown', component_property='value'),
        Input(component_id='age-dropdown', component_property='value'),
        Input(component_id='3DHR Slider4', component_property='value'),
        Input(component_id='3DHR Slider5', component_property='value'),
        Input(component_id='3DHR Slider6', component_property='value'),
        Input(component_id='trend-dropdown', component_property='value'),        

    ],
)
def bic_mape_2(selected_stat, selected_area, selected_age,p,d,q, trend):
    
    # Apply period index with year frequency to the Year feature 
    df['Year'] = pd.PeriodIndex(df['Year'], freq='Y').to_timestamp()
    df_ts = df[df['Statistic']==selected_stat]
    df_test = df_ts[df_ts['Area']==selected_area]
    df_test = df_test[df_test['Age Group']==selected_age]
    df_test = df_test.query("Sex=='Female'")

    df_test = df_test[['Year', 'VALUE']].groupby('Year', as_index=False).mean().set_index('Year')

    # Create a variable to get the forecast off the diagnostic results
    mod = sm.tsa.statespace.SARIMAX(df_test, 
                                    order=(p,d,q),
                                    trend = trend,
                                    enforce_stationarity=True,
                                    enforce_invertibility=False)

    results = mod.fit(max_iter=100, method='powell')

    bic = results.bic
    
    # Create a variable basen on the best model and start to predict from 1986
    pred = results.get_prediction(start = pd.to_datetime('2006-01-01'), 
                                       dynamic = False)


    predictions = pd.DataFrame(pred.predicted_mean)

    # Create a variable to get the forecast off the diagnostic results
    pred_uc = results.get_forecast(steps=5, dynamic=True)

    # Create the confidence interval variable
    pred_ci = pred_uc.conf_int(0.05) # Set confidence interval of 5%

    # Confidence interval equals to the pred_ci variable calculate in the built-in function provided by pdmarima
    ci = pred_ci

    # Forecast the future
    future = pred_uc.predicted_mean
    
    mape_female = MAPE(df_test, predictions.predicted_mean)
    
    fig_trace_01 = go.Indicator(
    mode = "number",
    #gauge = {'shape': "bullet"},
    value = bic,
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    title = {'text': "BIC"})
    
    fig_trace_02 = go.Indicator(
    mode = "number",
    #gauge = {'shape': "bullet"},
    value = mape_female,
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    title = {'text': "MAPE (%)"})
    
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{'type' : 'indicator'}, 
                {'type' : 'indicator'}]])
    
    fig.append_trace(fig_trace_01, row=1, col=1)
    fig.append_trace(fig_trace_02, row=1, col=2)
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0),
                      height=80,
                      title_font_color="#000000",
                      font_family="Arial", # Set Font style
                      font_size=10) # Set Font size) # legend false 

    return fig



# Callback & Function for Stats
@app.callback(
    Output(component_id='ts-graph-male', component_property='figure'),
    [
        Input(component_id='stats-dropdown', component_property='value'),
        Input(component_id='county-dropdown', component_property='value'),
        Input(component_id='age-dropdown', component_property='value'),
        Input(component_id='3DHR Slider', component_property='value'),
        Input(component_id='3DHR Slider2', component_property='value'),
        Input(component_id='3DHR Slider3', component_property='value'),
        Input(component_id='trend-dropdown', component_property='value'),        

    ],
)


def ts_plot_male(selected_stat, selected_area, selected_age,p,d,q, trend):
    

    # Apply period index with year frequency to the Year feature 
    df['Year'] = pd.PeriodIndex(df['Year'], freq='Y').to_timestamp()
    df_ts = df[df['Statistic']==selected_stat]
    df_test = df_ts[df_ts['Area']==selected_area]
    df_test = df_test[df_test['Age Group']==selected_age]
    df_test = df_test.query("Sex=='Male'")

    df_test = df_test[['Year', 'VALUE']].groupby('Year', as_index=False).mean().set_index('Year')

    # Create a variable to get the forecast off the diagnostic results
    mod = sm.tsa.statespace.SARIMAX(df_test, 
                                    order=(p,d,q),
                                    trend = trend,
                                    enforce_stationarity=True,
                                    enforce_invertibility=False)

    results = mod.fit(max_iter=100, method='powell')

    # Create a variable basen on the best model and start to predict from 1986
    pred = results.get_prediction(start = pd.to_datetime('2006-01-01'), 
                                       dynamic = False)


    predictions = pd.DataFrame(pred.predicted_mean)

    # Create a variable to get the forecast off the diagnostic results
    pred_uc = results.get_forecast(steps=5, dynamic=True)

    # Create the confidence interval variable
    pred_ci = pred_uc.conf_int(0.05) # Set confidence interval of 5%

    # Confidence interval equals to the pred_ci variable calculate in the built-in function provided by pdmarima
    ci = pred_ci

    # Forecast the future
    future = pred_uc.predicted_mean


    # Create Figure
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test.VALUE,
                        mode='lines',
                        marker=dict(color="#1E90FF"),
                        line=dict(color='#1E90FF', width=4), 
                        name='Observed'))

    fig.add_trace(go.Scatter(x=predictions.index, y=predictions.predicted_mean,
                        mode='lines+markers',
                        marker=dict(color="#000000"),
                        line=dict(color='#000000', width=1, dash='dot'),
                        name='Predictions')),

    fig.add_trace(go.Scatter(x=future.index, y=future,
                        mode='lines+markers',
                        marker=dict(color="#090059"),
                        line=dict(color='#090059', width=1, dash='dot'),
                        name='Forecast')),

    # Add the Confidence Interval for the Lower Bounds on the test test
    fig.add_trace(go.Scatter(x=ci.index, y=ci["lower VALUE"],
                        marker=dict(color="lightblue"),
                        line=dict(width=0),
                        mode='lines',
                        fillcolor='rgba(32, 146, 230,0.3)',
                        fill='tonexty',
                        name='Lower Bound'))

    # Add the Confidence Interval for the Upper Bounds on the test test
    fig.add_trace(go.Scatter(x=ci.index, y=ci["upper VALUE"],
                        marker=dict(color="blue"),
                        line=dict(width=0),
                        mode='lines',
                        fillcolor='rgba(32, 146, 230,0.3)',
                        fill='tonexty',
                        name='Upper Bound'))



    # Use update_layout in order to define few configuration such as figure height and width, title, etc
    fig.update_layout(
        title = 'Male',
        title_font_color="#000000",
        height=400, # Figure height
        width=700, # Figure width
        showlegend=False,
        font_family="Arial", # Set Font style
        font_size=14,
        hovermode="x",
        template = 'none',
        margin=dict(l=50, r=0, t=50, b=20))

    # Add Spikes
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)

    # Show Figure
    return fig


# Callback & Function for Stats
@app.callback(
    Output(component_id='ts-graph-female', component_property='figure'),
    [
        Input(component_id='stats-dropdown', component_property='value'),
        Input(component_id='county-dropdown', component_property='value'),
        Input(component_id='age-dropdown', component_property='value'),
        Input(component_id='3DHR Slider4', component_property='value'),
        Input(component_id='3DHR Slider5', component_property='value'),
        Input(component_id='3DHR Slider6', component_property='value'),
        Input(component_id='trend-dropdown', component_property='value'),        

    ],
)


def ts_plot_female(selected_stat, selected_area, selected_age,p,d,q, trend):
    

    # Apply period index with year frequency to the Year feature 
    df['Year'] = pd.PeriodIndex(df['Year'], freq='Y').to_timestamp()
    df_ts = df[df['Statistic']==selected_stat]
    df_test = df_ts[df_ts['Area']==selected_area]
    df_test = df_test[df_test['Age Group']==selected_age]
    df_test = df_test.query("Sex=='Female'")

    df_test = df_test[['Year', 'VALUE']].groupby('Year', as_index=False).mean().set_index('Year')

    # Create a variable to get the forecast off the diagnostic results
    mod = sm.tsa.statespace.SARIMAX(df_test, 
                                    order=(p,d,q),
                                    trend = trend,
                                    enforce_stationarity=True,
                                    enforce_invertibility=False)

    results = mod.fit(max_iter=100, method='powell')

    # Create a variable basen on the best model and start to predict from 1986
    pred = results.get_prediction(start = pd.to_datetime('2006-01-01'), 
                                       dynamic = False)


    predictions = pd.DataFrame(pred.predicted_mean)

    # Create a variable to get the forecast off the diagnostic results
    pred_uc = results.get_forecast(steps=5, dynamic=True)

    # Create the confidence interval variable
    pred_ci = pred_uc.conf_int(0.05) # Set confidence interval of 5%

    # Confidence interval equals to the pred_ci variable calculate in the built-in function provided by pdmarima
    ci = pred_ci

    # Forecast the future
    future = pred_uc.predicted_mean


    # Create Figure
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test.VALUE,
                        mode='lines',
                        marker=dict(color="#997070"),     
                        line=dict(color='#997070', width=4),
                        name='Observed'))

    fig.add_trace(go.Scatter(x=predictions.index, y=predictions.predicted_mean,
                        mode='lines+markers',
                        marker=dict(color="#000000"),
                        line=dict(color='#000000', width=1, dash='dot'),
                        name='Predictions')),

    fig.add_trace(go.Scatter(x=future.index, y=future,
                        mode='lines+markers',
                        marker=dict(color="#090059"),
                        line=dict(color='#090059', width=1, dash='dot'),
                        name='Forecast')),

    # Add the Confidence Interval for the Lower Bounds on the test test
    fig.add_trace(go.Scatter(x=ci.index, y=ci["lower VALUE"],
                        marker=dict(color="lightblue"),
                        line=dict(width=0),
                        mode='lines',
                        fillcolor='rgba(32, 146, 230,0.3)',
                        fill='tonexty',
                        name='Lower Bound'))

    # Add the Confidence Interval for the Upper Bounds on the test test
    fig.add_trace(go.Scatter(x=ci.index, y=ci["upper VALUE"],
                        marker=dict(color="blue"),
                        line=dict(width=0),
                        mode='lines',
                        fillcolor='rgba(32, 146, 230,0.3)',
                        fill='tonexty',
                        name='Upper Bound'))



    # Use update_layout in order to define few configuration such as figure height and width, title, etc
    fig.update_layout(
        title = 'Female',
        title_font_color="#000000",
        height=400, # Figure height
        width=700, # Figure width
        showlegend=False,
        font_family="Arial", # Set Font style
        font_size=14,
        hovermode="x",
        template = 'none',
        margin=dict(l=50, r=0, t=50, b=20))

    # Add Spikes
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)

    # Show Figure
    return fig




if __name__ == '__main__':
    app.run_server(debug=False,host = '127.0.0.1')
