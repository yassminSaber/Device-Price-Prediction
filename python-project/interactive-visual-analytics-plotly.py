import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Assume 'df' is your DataFrame containing the dataset
# Replace 'df' with your actual DataFrame name

# Initialize Dash app
app = dash.Dash(__name__)

df = pd.read_csv("train_data_2.csv")

# Define app layout
app.layout = html.Div([
    dcc.Dropdown(
        id='feature-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns[:-1]],  # Exclude target column
        value=df.columns[0]  # Default feature selection
    ),
    dcc.Graph(id='feature-price-plot')
])

# Define callback to update graph based on selected feature
@app.callback(
    Output('feature-price-plot', 'figure'),
    [Input('feature-dropdown', 'value')]
)
def update_plot(selected_feature):
    fig = px.scatter(df, y=selected_feature, x='price_range', color='price_range',
                     labels={'price_range': 'Price Range'}, title=f'Relationship between {selected_feature} and Price Range')
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server()
