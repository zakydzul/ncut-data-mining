import json
import dash
import plotly.graph_objects as go
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import copy
import os
from sklearn.linear_model import LinearRegression
import plotly.express as px
import numpy as np

#---------init app-------------#
app = dash.Dash(__name__,
                )
server = app.server

#---------open assets and data-------------#
with open('assets/indonesia-newprovgeo.json') as response:
    provincies = json.load(response)

df_ina = pd.read_csv("data/Indonesia_statistics_2010_2019_fixed.csv",
                   dtype={"Year": str,})

df = pd.read_csv("data/province list.csv")

file_pattern=(
    'province GDP.csv',
    'province college rate.csv',
    'province economic growth.csv',
    'province literacy rate.csv',
    'province population.csv',
)
main_path = 'data'

for i,x in enumerate(file_pattern):
    temp = pd.read_csv(os.path.join(main_path,x))
    df = pd.concat([df, temp], axis=1)

#---------Figure Plotting----------#



# ---------app layout-------------#
app.layout = html.Div(
    [dbc.Card(dbc.CardBody([
        dbc.Row([#Header Start Here
            dbc.Col([
                html.Div(
                    [#logo begins
                        html.Img(
                            src=app.get_asset_url('NCUT.png'),
                            id='NCUT logo',
                            style={
                                'height' : "auto",
                                'width': '90px',
                            },
                        )
                    ],#logo ends
                ),
            ], width='auto'),
            dbc.Col([
                html.Div(
                    [#Main Title begins
                        html.H4(
                            'NCUT Data Mining Final Project',
                            style={
                                'height':'auto',
                                'width' :'100%',
                                'font-family':'roboto',
                                "margin-left": "10px",
                                },
                        ),
                        html.P(
                            'This app is built to fulfill final exam in Data Mining Class at NCUT Taiwan.',
                            style={
                                'font-family':'roboto',
                                "margin-left": "10px",
                                'margin-top':'0px',
                                'margin-bottom':'0px',}
                        ),
                        html.P(
                            'Created by: Zaky Dzulfikri, 2021',
                                style={
                                'font-family':'roboto',
                                "margin-left": "10px",
                                'margin-top':'0px',
                                'margin-bottom':'0px',}
                        ),
                    ],
                ),
            ]),
        ],align='center' ,style={'margin-bottom': '20px'}, no_gutters=True,), #Main Title ends
        # infographics title Starts
        dbc.Row(
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row(dbc.Col([
                                html.Div([
                                html.H1(
                                    'INDONESIA INFOGRAPHICS',
                                    ),],style={'text-align':'center',}, className='text-black-50',
                                ),
                        ])),
                        dbc.Row([
                            dbc.Col([
                                dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
                                    html.Div(
                                        [html.H2('34'), html.H4("provinces")],
                                        style={'text-align':'center',},
                                    ),
                                ]),className='text-white bg-primary mb-3'))),
                                dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
                                    html.Div(
                                        [html.H2('1,916,906.77'),
                                         html.H4("sq.km of area")],
                                        style={'text-align':'center',},
                                    ),
                                ]),className='text-white bg-primary mb-3'))),
                                dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
                                        html.Div(
                                            [html.H2('16,056'), html.H4("Islands")],
                                            style={'text-align': 'center', },
                                        ),
                                ]),className='text-white bg-primary mb-3'))),
                            ],width=2),
                            dbc.Col([
                                html.Div(
                                    [#Second Column Starts
                                        html.Img(src=app.get_asset_url('peta_indonesia.png'),
                                                 style={
                                                'height' : "auto",
                                                'width': '100%',
                                                 },
                                        ),
                                    ],#Second Column ends
                                ),
                            ],width=8),
                            dbc.Col([
                                dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
                                    html.Div(
                                        [html.H1('267.7'), html.H5("million citizen")],
                                        style={'text-align': 'center', },
                                    ),
                                ]),className='text-white bg-primary mb-3'))),
                                dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
                                    html.Div(
                                        [html.H5("More than"),
                                         html.H2('1000'),
                                         html.H5("Local language")],
                                        style={'text-align': 'center', },
                                    ),
                                ]),className='text-white bg-primary mb-3'))),
                                dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
                                        html.Div(
                                            [html.H2('130'), html.H5("Million Facebook users")],
                                            style={'text-align': 'center', },
                                        ),
                                ]),className='text-white bg-primary mb-3'))),
                            ],width=2),
                        ])
                    ])
                ],className='bg-light mb-3', style={'margin':'20'}),
                width=12,
            ),
        ),#infographic ends
        dbc.Row(dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([#Row Title
                       dbc.Col([
                            html.Div([
                                html.H2('Statistics',),
                            ],style={'text-align':'center',}, className='text-black-50',),
                       ]),
                    ]),
                    dbc.Row([#Row Display Graph
                        dbc.Col([#Column Social
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            dbc.Row(#Card Explaination
                                                dbc.Col([
                                                    html.Div([
                                                        html.H3('Indonesia Economic and Social Statistics'),
                                                        html.P('Select different data,  there are eight data to choose, '
                                                               'the data is ranging from 2010 to 2019.'),
                                                    ]),
                                                ])
                                            ),
                                            dbc.Row(
                                                dbc.Col([
                                                    html.Div([
                                                        dcc.Dropdown(
                                                            id='data_sel',
                                                            options=[
                                                                {'label': 'Population (Million)','value':'Population'},
                                                                {'label': 'Life Expectancy Rate (Years)','value':'Life expectancy rate'},
                                                                {'label': 'Literacy Rate (%)','value':'Literacy Rate'},
                                                                {'label': 'Poor People (Million)','value':'Poor People'},
                                                                {'label': 'Human Development Index','value':'Human Development Index'},
                                                                {'label': 'Gross Domestic Product (Trillion Rupiah)','value':'GDP'},
                                                                {'label': 'Economic Growth (%)','value':'Economic Growth'},
                                                                {'label': 'Per capita of GDP (Million Rupiah)','value':'Per capita of GDP'},
                                                            ],
                                                            value='Population',
                                                            multi=False,
                                                            className='text-primary',
                                                        )
                                                    ])
                                                ])
                                            ),
                                            dbc.Row(
                                                dbc.Col([
                                                    html.Div([
                                                        html.P('Data is taken from Statistics Indonesia '
                                                               '(Badan Pusat Statistik)')
                                                    ], style={'margin-top':'10px'}),
                                                    dcc.Markdown(
                                                        children=[
                                                            'Source: [Badan Pusat Statistik] (https://www.bps.go.id/)'
                                                        ]
                                                    )
                                                ])
                                            ),
                                        ])
                                    ],className='text-white bg-primary mb-3',style={'height':'492px'})
                                ],width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            dbc.Row([
                                                dbc.Col([
                                                    html.Div([
                                                        dcc.Graph(id='social_graph'),
                                                    ]),
                                                ]),
                                            ]),
                                        ],),
                                    ],className='text-white bg-primary mb-3'),
                                ], width=9),
                            ]),
                        ],),
                    ],),
                    dbc.Row([#Row Map
                        dbc.Col([#Column Social
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            dbc.Row(#Card Explaination
                                                dbc.Col([
                                                    html.Div([
                                                        html.H3('Indonesia Provinces Statistics'),
                                                        html.P('Select different data,  there are eight data to choose')
                                                    ]),
                                                ])
                                            ),
                                            dbc.Row(
                                                dbc.Col([
                                                    html.Div([
                                                        dcc.Dropdown(
                                                            id='data_sel_map',
                                                            options=[
                                                                {'label': 'Population (Million)','value':'Population'},
                                                                {'label': 'Literacy Rate (%)','value':'Literacy Rate'},
                                                                {'label': 'Gross Domestic Product (Trillion Rupiah)','value':'GDP'},
                                                                {'label': 'Economic Growth (%)','value':'Economic Growth'},
                                                                {'label': 'College Rate (%)','value':'College Rate'},
                                                            ],
                                                            value='Population',
                                                            multi=False,
                                                            className='text-primary',
                                                        )
                                                    ])
                                                ])
                                            ),
                                            dbc.Row(
                                                dbc.Col([
                                                    html.Div([
                                                        html.P('Data is taken from Statistics Indonesia '
                                                               '(Badan Pusat Statistik)')
                                                    ], style={'margin-top':'10px'}),
                                                    dcc.Markdown(
                                                        children=[
                                                            'Source: [Badan Pusat Statistik] (https://www.bps.go.id/)'
                                                        ]
                                                    )
                                                ])
                                            ),
                                        ])
                                    ],className='text-white bg-primary mb-3',style={'height':'492px'})
                                ],width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            dbc.Row([
                                                dbc.Col([
                                                    html.Div([
                                                        dcc.Graph(id='map'),
                                                    ]),
                                                ]),
                                            ]),
                                        ],),
                                    ],className='text-white bg-primary mb-3'),
                                ], width=9),
                            ]),
                        ],),
                    ],),
                ],),
            ],className='bg-light mb-3',),
        ),),
        dbc.Row(dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row(dbc.Col([
                        html.Div([
                            html.H2('Data Regression', ),
                        ], style={'text-align': 'center', }, className='text-black-50', ),
                    ],),),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                dcc.Graph(id='regression', style={'height':'600px'})
                            ])
                        ],width=12),
                    ],),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H5('Select X axis', className='text-black-50',),
                                dcc.Dropdown(
                                    id='data_sel_x',
                                    options=[
                                        {'label': 'Population (Million)', 'value': 'Population'},
                                        {'label': 'Life Expectancy Rate (Years)', 'value': 'Life expectancy rate'},
                                        {'label': 'Literacy Rate (%)', 'value': 'Literacy Rate'},
                                        {'label': 'Poor People (Million)', 'value': 'Poor People'},
                                        {'label': 'Human Development Index', 'value': 'Human Development Index'},
                                        {'label': 'Gross Domestic Product (Trillion Rupiah)', 'value': 'GDP'},
                                        {'label': 'Economic Growth (%)', 'value': 'Economic Growth'},
                                        {'label': 'Per capita of GDP (Million Rupiah)', 'value': 'Per capita of GDP'},
                                    ],
                                    value='Population',
                                    multi=False,
                                    className='text-primary',
                                )
                            ]),
                        ],width=4, style={'margin-top':'10px'}),
                        dbc.Col([
                            html.Div([
                                html.H5('Select Y axis', className='text-black-50',),
                                dcc.Dropdown(
                                    id='data_sel_y',
                                    options=[
                                        {'label': 'Population (Million)', 'value': 'Population'},
                                        {'label': 'Life Expectancy Rate (Years)', 'value': 'Life expectancy rate'},
                                        {'label': 'Literacy Rate (%)', 'value': 'Literacy Rate'},
                                        {'label': 'Poor People (Million)', 'value': 'Poor People'},
                                        {'label': 'Human Development Index', 'value': 'Human Development Index'},
                                        {'label': 'Gross Domestic Product (Trillion Rupiah)', 'value': 'GDP'},
                                        {'label': 'Economic Growth (%)', 'value': 'Economic Growth'},
                                        {'label': 'Per capita of GDP (Million Rupiah)', 'value': 'Per capita of GDP'},
                                    ],
                                    value='Population',
                                    multi=False,
                                    className='text-primary',
                                )
                            ]),
                        ],width=4, style={'margin-top':'10px'}),
                    ]),
                ]),
            ],className='bg-light mb-3',),
        ),),
    ]),className='text-white bg-primary mb-3',)
    ],id="mainContainer",
)

layout = dict(
    autosize=True,
    #automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview",
    xaxis_title='Years',
    yaxis_title='Million'
)
label_dict = {
    'Population':'Million',
    'Life expectancy rate':'Years',
    'Literacy Rate':'%',
    'Poor People':'Million',
    'Human Development Index':'Units',
    'GDP':'Trillion Rupiahs',
    'Economic Growth':'%',
    'Per capita of GDP':'Million Rupiahs',
}
#---------app call back----------#
@app.callback(
    Output('social_graph','figure'),
    [Input('data_sel','value')]
)
def update_graph(pop):
    indiv_layout = copy.deepcopy(layout)
    dff_ina = df_ina.copy()
    indiv_layout['title'] = pop
    indiv_layout['yaxis_title'] = label_dict[pop]
    data=[
        dict(
            type="scatter",
            mode="lines+markers",
            name=pop,
            x=dff_ina['Year'],
            y=dff_ina[pop],
            line=dict(shape="spline", smoothing=2, width=1, color="#2c3e50"),
            marker=dict(symbol="diamond-open"),),
        ]
    fig = dict(data=data,
               layout=indiv_layout
               )
    return fig

@app.callback(
    Output('map','figure'),
    [Input('data_sel_map','value')]
)
def update_map(val):
    fig = go.Figure(go.Choroplethmapbox(geojson=provincies, locations=df['prov'], z=df[val],
                                        featureidkey="properties.Propinsi",
                                        colorscale="Viridis", zmin=min(df[val]), zmax=max(df[val]),
                                        marker_opacity=0.5, marker_line_width=0))
    fig.update_layout(mapbox_style="carto-positron",
                      mapbox_zoom=4.2, mapbox_center={"lat": -3.0893, "lon": 115.9213})
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, )

    return fig

@app.callback(
    Output('regression','figure'),
    [Input('data_sel_x','value'),
     Input('data_sel_y','value'),]
)
def update_regression(x,y):
    dff_x = df_ina.copy()
    dff_y = df_ina.copy()
    x_data = np.asarray(dff_x[x]).reshape(-1,1)
    model = LinearRegression()
    model.fit(x_data,dff_y[y])

    x_range = np.linspace(x_data.min(),x_data.max(), 100)
    y_range = model.predict(x_range.reshape(-1,1))

    fig  = px.scatter(dff_x, x=x, y=y, opacity=0.65)
    fig.add_trace(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))

    return fig

#---------run server-------------#
if __name__ == '__main__':
    app.run_server(debug=True)