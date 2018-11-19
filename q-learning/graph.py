import plotly as py
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import pandas
py.offline.init_notebook_mode(connected=True)


def graph(dataframe_set, color_set, mode_set, name):
    data = []
    x_axis_label = ''
    y_axis_label = ''
    for dataframe, color, mode in zip(dataframe_set, color_set, mode_set):
        x_axis_label = dataframe.columns[0]
        y_axis_label = dataframe.columns[1]
        data.append(
            go.Scatter(
                x=dataframe[x_axis_label],
                y=dataframe[y_axis_label],
                mode= mode,
                name=name,
                marker = dict(
                    color = color,
                    line = dict(width = 1)
                )
            )
        )

    fig= {
        'data': data,
        'layout': {
            'xaxis': {'title': x_axis_label},
            'yaxis': {'title': y_axis_label}
        }
    }

    py.offline.plot(fig,filename=name)
