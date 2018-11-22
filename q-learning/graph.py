import plotly as py
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import pandas
py.offline.init_notebook_mode(connected=True)

def map_to_graph(dataframe_color, color_set):
    color_to_map = []
    for index, row in dataframe_color.iterrows():
        if row[1] == 1:
            # green
            color_to_map.append(color_set[0])
        elif row[1] == -1:
            # red
            color_to_map.append([int(row[0]), color_set[1]])
    return color_to_map

def graph(dataframe_set, color_scale, mode_set, name):
    data = []
    x_axis_label = ''
    y_axis_label = ''
    for dataframe, mode in zip(dataframe_set, mode_set):
        x_axis_label = dataframe.columns[0]
        y_axis_label = dataframe.columns[1]
        data.append(
            go.Scatter(
                x=dataframe[x_axis_label],
                y=dataframe[y_axis_label],
                mode=mode,
                name=name,
                marker = dict(
                    size=1,
                    color = color_scale,
                    showscale=False
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
