import plotly as py
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import pandas as pd
py.offline.init_notebook_mode(connected=True)

def map_to_graph(dataframe_color,dataframe_set, color_set):
    buy_to_map = []
    sell_to_map = []
    for row_color, row_set in zip(dataframe_color.iterrows(),dataframe_set.iterrows()):
        if row_color[1][1] == 1:
            # green
            buy_to_map.append([float(row_set[1][0]),float(row_set[1][1])])
        elif row_color[1][1] == -1:
            # red
            sell_to_map.append([float(row_set[1][0]),float(row_set[1][1])])

    return pd.DataFrame(buy_to_map), pd.DataFrame(sell_to_map)

def graph(dataframe_set, buy_to_map, sell_to_map, name):
    data = []
    x_axis_label = ''
    y_axis_label = ''
    for dataframe, buy_to_map, sell_to_map in zip(dataframe_set, buy_to_map, sell_to_map):
        x_axis_label = dataframe.columns[0]
        y_axis_label = dataframe.columns[1]
        data.append(
            go.Scatter(
                x=dataframe[x_axis_label],
                y=dataframe[y_axis_label],
                mode='lines+markers',
                name='Buy/Sell',
                marker = dict(
                    size=1,
                    color = 'rgb(0,0,0)'
                )
            )
        )
        if not buy_to_map.empty:
            buy_to_map_x_axis_label = buy_to_map.columns[0]
            buy_to_map_y_axis_label = buy_to_map.columns[1]
            data.append(go.Scatter(
                x=buy_to_map[buy_to_map_x_axis_label],
                y=buy_to_map[buy_to_map_y_axis_label],
                mode='markers',
                name='Buy',
                marker = dict(
                    size=5,
                    color = 'rgb(0,128,0)'
                )
            ))

        if not sell_to_map.empty:
            sell_to_map_x_axis_label = sell_to_map.columns[0]
            sell_to_map_y_axis_label = sell_to_map.columns[1]
            data.append(go.Scatter(
                x=sell_to_map[sell_to_map_x_axis_label],
                y=sell_to_map[sell_to_map_y_axis_label],
                mode='markers',
                name='Sell',
                marker = dict(
                    size=5,
                    color = 'rgb(255,0,0)'
                )
            ))
    fig= {
        'data': data,
        'layout': {
            'xaxis': {'title': x_axis_label},
            'yaxis': {'title': y_axis_label}
        }
    }

    py.offline.plot(fig,filename=name)
