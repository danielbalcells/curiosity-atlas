import pandas as pd
import itertools

import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc

from sklearn.neighbors import NearestNeighbors
import seaborn as sns



def serve_layout():
    display_tags = tags_df[tags_df['count'] > 1]
    display_tag_labels = list(display_tags.apply(lambda x: f"{x['name']} ({x['count']})", axis=1))
    display_tag_names = display_tags['name']
    return html.Div(
        children=[
            dcc.Graph(
                figure=update_graph([]),
                id='graph-content',
                style={
                    'width': '80%',
                    'height': '100vh',
                    'overflow': 'auto'
                },
                config={
                    'displayModeBar': False
                }
            ),
            html.Div(
                children=[
                    dcc.Checklist(
                        id='tag-selector',
                        options=[{'label': html.Span(
                            [html.Div(style={'background': color_by_tag[tag],
                                             'border-radius': '50%',
                                             'width': '10px',
                                             'height': '10px',
                                             'display': 'inline-block',
                                             'margin-right': '5px'}), label], 
                                style={'display': 'inline-block'}), 'value': tag} for label, tag in
                                 zip(display_tag_labels, display_tag_names)],
                        value=[]
                    )
                ],
                id='controls',
                style={
                    'width': '20%',
                    'height': '100vh',
                    'overflow': 'auto',
                    'padding': '20px'
                }
            )
        ],
        style={'display': 'flex'}
    )

@callback(
    Output('graph-content', 'figure'),
    Input('tag-selector', 'value')
)
def update_graph(selected_tags):
    lines_by_tag = get_lines_by_tag(tools, selected_tags)
    fig = go.Figure(layout={'template': 'plotly_dark'})

    for tag, lines in lines_by_tag.items():
        if tag not in selected_tags:
            for line in lines:
                fig.add_trace(line)

    for tag, lines in lines_by_tag.items():
        if tag in selected_tags:
            for line in lines:
                fig.add_trace(line)

    scatter = go.Scatter(x=tools.vis_dim1, y=tools.vis_dim2,
                         mode='markers',
                         text=tools.hover_text,
                         customdata=tools.hover_text,
                         hovertemplate='%{customdata}<extra></extra>',
                         marker=dict(
                             color='white',
                             size=15
                         )
                         )

    fig.add_trace(scatter)

    fig.update_traces(textposition='top center')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        font=dict(
            size=11,  # Set the font size here
        )
    )
    return fig

def get_lines_by_tag(tools, selected_tags):
    lines_by_tag = {tag: [] for tag in tags}

    for tag in tags:
        tag_color = color_by_tag[tag] if tag in selected_tags else 'gray'
        tag_width = 4 if tag in selected_tags else 1
        tag_tools = tools[tools.Tags.str.contains(tag)]
        if len(tag_tools) == 1:
            continue
        elif len(tag_tools) == 2:
            knn = NearestNeighbors(n_neighbors=2)
        else:
            knn = NearestNeighbors(n_neighbors=3)
        knn.fit(tag_tools[['vis_dim1', 'vis_dim2']].values)

        for _, tool in tag_tools.iterrows():
            distances, indices = knn.kneighbors([[tool['vis_dim1'], tool['vis_dim2']]])

            for neighbor_index in indices[0][1:]:
                neighbor = tag_tools.iloc[neighbor_index]
                line = go.Scatter(
                        x=[tool['vis_dim1'], neighbor['vis_dim1']],
                        y=[tool['vis_dim2'], neighbor['vis_dim2']],
                        mode='lines',
                        line=dict(
                            width=tag_width,
                            color=tag_color
                        ),  # Visible line
                        hoverinfo='text',
                        text=[tag, tag],  # Hover text is the tag name
                        hoverlabel=dict(bgcolor=tag_color)  # Hover label color matches the line color
                    )
                lines_by_tag[tag].append(line)
    return lines_by_tag


def get_color_by_tag(tags):
    small_palette = sns.color_palette("hls", 10).as_hex()
    palette_cycle = itertools.cycle(small_palette)
    color_by_tag = {tag: next(palette_cycle) for tag in tags}
    return color_by_tag

def get_tags(tools):
    tags = []
    for i, tool in tools.iterrows():
        tags += [t for t in str.split(tool.Tags, ', ') if t not in tags]
    tags_df = pd.DataFrame(
        {'name': tags, 
         'count': [tools.apply(lambda x: tag in x.Tags, axis=1).sum()
                             for tag in tags]})
    tags_df.sort_values(by='count', ascending=False, inplace=True)
    return tags_df, list(tags_df.name)

def load_data(tools_file):
    tools = pd.read_json(tools_file)
    tags_df, tags = get_tags(tools)
    color_by_tag = get_color_by_tag(tags)
    return tools, tags_df, tags, color_by_tag

tools_file = ("./tools_embeddings_23-7-20.json")
tools, tags_df, tags, color_by_tag = load_data(tools_file)

app = Dash(__name__, external_stylesheets=[dbc.themes.QUARTZ])

app.layout = serve_layout()

if __name__ == '__main__':
    app.run(debug=True)
