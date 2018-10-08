import plotly as py
import cufflinks as cf
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
py.offline.init_notebook_mode(connected=True)

import pandas

def graph(dataframe_set,name):
    data = []
    for dataframe in dataframe_set:
        x_axis_label = dataframe.columns[0]
        y_axis_label = dataframe.columns[1]
        data.append(
            go.Scatter(
                x = dataframe[x_axis_label],
                y = dataframe[y_axis_label],
                mode = 'markers',
                name = name
            )
        )

    fig= {
        'data': data,
        'layout':{
            'xaxis': {'title': x_axis_label},
            'yaxis': {'title': y_axis_label}
        }
    }

    py.offline.plot(fig,filename=name)
