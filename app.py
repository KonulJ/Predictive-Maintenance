import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.express as px
import io
import base64


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Load datasets
df_train = pd.read_csv('df_train.csv')
df_valid = pd.read_csv('df_valid.csv')
df_train_eda = pd.read_csv('df_train_eda.csv')
df_valid_eda = pd.read_csv('df_valid_eda.csv')
df_train_fe = pd.read_csv('df_train_fe.csv')
df_valid_fe = pd.read_csv('df_valid_fe.csv')
df_train_fe_dc = pd.read_csv('df_train_fe_dc.csv')
df_valid_fe_dc = pd.read_csv('df_valid_fe_dc.csv')
df_train_fe_nr_n = pd.read_csv('df_train_fe_nr_n.csv')
df_valid_fe_nr_n = pd.read_csv('df_valid_fe_nr_n.csv')
df_truth = pd.read_csv('y_valid.csv')
average_lifecycles = pd.read_csv('average_lifecycles.csv') 

# Define column names for df_train
df_train_columns = ['engine_id', 'cycles', 'op_set_1', 'op_set_2', 'op_set_3'] + [f's{i}' for i in range(1, 22)] + ['RUL']
df_train.columns = df_train_columns


# Sensor dictionary
sensor_dictionary = {
    's2': '(LPC outlet temperature) (◦R)',
    's3': '(HPC outlet temperature) (◦R)',
    's4': '(LPT outlet temperature) (◦R)',
    's7': '(HPC outlet pressure) (psia)',
    's8': '(Physical fan speed) (rpm)',
    's9': '(Physical core speed) (rpm)',
    's11': '(HPC outlet Static pressure) (psia)',
    's12': '(Ratio of fuel flow to Ps30) (pps/psia)',
    's13': '(Corrected fan speed) (rpm)',
    's14': '(Corrected core speed) (rpm)',
    's15': '(Bypass Ratio)',
    's17': '(Bleed Enthalpy)',
    's20': '(High-pressure turbines Cool air flow)',
    's21': '(Low-pressure turbines Cool air flow)'
}

def generate_rul_distribution_plot():
    fig = px.histogram(df_train_eda, x='RUL', nbins=5, title='Distribution of Remaining Useful Life (RUL)')
    fig.update_layout(
        xaxis_title='Remaining Useful Life',
        yaxis_title='Frequency',
        bargap=0.2,
        template='plotly_white'
    )
    return fig


# Dropdown options
dataset_options = {
    "df_train_eda": "Non-Preprocessed Dataset",
    "df_train_fe": "Min-Max Normalization",
    "df_train_fe_nr_n": "First N-Value Normalization"
}

sensor_columns = [col for col in df_train_eda.columns if col.startswith('s')]
engine_ids = sorted(df_train_eda['engine_id'].unique())

# Function to generate time series plot
def generate_time_series_plot(dataset_name, engine_id, sensor_name):
    dataset = globals()[dataset_name]  # Access dataset dynamically
    df = dataset[dataset['engine_id'] == engine_id]

    plt.figure(figsize=(10, 5))
    plt.plot(df['time_in_cycles'], df[sensor_name], marker='o', linestyle='-')
    plt.xlabel('Cycle')
    plt.ylabel(sensor_name)
    plt.title(f'Engine {engine_id} - {sensor_name}')
    plt.grid(True)
    return save_plot()


# Function to generate sensor plots
def generate_sensor_plot(sensor_name):
    fig = go.Figure()

    # Select specific engines for visualization
    selected_engines = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for engine_id in selected_engines:
        engine_data = df_train_fe_dc[df_train_fe_dc['engine_id'] == engine_id].copy()

        # Apply rolling mean before plotting
        engine_data[f"{sensor_name}_smooth"] = engine_data[sensor_name].rolling(10, min_periods=1).mean()

        fig.add_trace(go.Scatter(
            x=engine_data['RUL'],
            y=engine_data[f"{sensor_name}_smooth"],  
            mode='lines',
            name=f'Engine {engine_id}'
        ))

    fig.update_layout(
        title=f'Evolution of {sensor_name}: {sensor_dictionary.get(sensor_name, "Unknown Sensor")}',
        xaxis_title='Remaining Useful Life',
        yaxis_title=sensor_dictionary.get(sensor_name, "Sensor Reading"),
        xaxis=dict(autorange="reversed"),  
        legend_title="Engine ID"
    )

    return fig

# Function to generate bias plot
def generate_bias_plot():
    fig = go.Figure(data=[go.Bar(x=df_performance['Iteration'], y=df_performance['Bias'], marker_color='orange')])
    fig.update_layout(title="Bias", xaxis_title="Iteration", yaxis_title="Bias")
    return fig

# Sample Model Performance Data
performance_data = [
    {'Iteration': 'Iteration 1', 'MAE': 27.19, 'MSE': 1163.37, 'RMSE': 34.11, 'R²_Score': 0.33,'Bias': -21.13},
    {'Iteration': 'Iteration 2', 'MAE': 18.20, 'MSE': 786.76, 'RMSE': 28.05, 'R²_Score': 0.54,'Bias': 7.91},
    {'Iteration': 'Iteration 3', 'MAE': 30.99, 'MSE': 1419.32, 'RMSE': 37.67, 'R²_Score': 0.18,'Bias':-29.14},
    {'Iteration': 'Iteration 4', 'MAE': 11.14, 'MSE': 233.45, 'RMSE': 15.28, 'R²_Score': 0.86,'Bias': 15.55}
]

df_performance = pd.DataFrame(performance_data)




# Function to save plot
def save_plot():
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded_image = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    return f"data:image/png;base64,{encoded_image}"

server = app.server 
app.layout = dbc.Container([
    # Project Title
    dbc.Row([dbc.Col(html.H1("Project Dashboard"), className="mb-4")]),
    
    # ======== INTRODUCTION SECTION ======== 
    dbc.Row([dbc.Col(html.H2("Introduction", className="mb-3"))]),
    dbc.Row([
        dbc.Col(html.P(
            "In today’s fast-paced industrial landscape, equipment failures can have severe financial and operational consequences. "
            "Unplanned downtime results in significant revenue loss, increased maintenance expenses, and potential safety hazards. "
            "Traditional maintenance strategies—such as reactive maintenance, where repairs are conducted only after a failure occurs, "
            "and preventive maintenance, where maintenance is performed on a fixed schedule—often lead to inefficiencies, excessive costs, and unnecessary servicing."
        ))
    ]),
    dbc.Row([
        dbc.Col(html.P(
            "Predictive Maintenance (PdM) offers a data-driven approach to solving these challenges. By leveraging machine learning and advanced analytics, "
            "PdM predicts equipment failures before they happen, allowing businesses to schedule maintenance only when needed. This transition from time-based "
            "to condition-based maintenance not only optimizes resources and reduces downtime but also extends the lifespan of machinery, leading to higher "
            "operational efficiency and cost savings."
        ))
    ]),
    dbc.Row([
        dbc.Col(html.P(
            "This project utilizes the C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset, a widely used benchmark dataset for predictive maintenance. "
            "The dataset consists of simulated aircraft engine sensor readings collected under different operating conditions and fault scenarios. "
            "It contains multiple operational settings and sensor measurements, which help in understanding engine degradation over time."
        ))
    ]),
    
    # ======== BUSINESS OBJECTIVE SECTION ========
    dbc.Row([dbc.Col(html.H2("Business Objective", className="mb-3 text-right"))]),  
    dbc.Row([
        dbc.Col(html.P(
            "The primary goal of this project is to develop an intelligent Predictive Maintenance (PdM) system using the C-MAPSS dataset to accurately predict potential equipment failures before they occur. "
            "By leveraging simulated sensor readings under different operational settings, this system provides actionable insights for optimizing maintenance schedules and minimizing operational disruptions."
        ), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.P("Through machine learning models, this project aims to:"), width=12, className="text-right")
    ]),
    dbc.Row([
        dbc.Col(html.Ul([
        
            html.Li("Accurately predict failures by analyzing sensor data across varying engine operating conditions."),
            html.Li("Optimize maintenance schedules by forecasting failures and recommending timely interventions."),
            html.Li("Reduce operational costs by preventing unexpected failures that lead to expensive emergency repairs."),
            html.Li("Minimize unplanned downtime, ensuring higher equipment availability and productivity."),
            html.Li("Extend engine lifespan through condition-based maintenance, reducing unnecessary servicing."),
            html.Li("Improve safety by detecting early signs of failure, reducing risks of catastrophic breakdowns.")
        ]), width={"size": 8, "offset": 2})
    ]),
    dbc.Row([
        dbc.Col(html.P(
            "The use of C-MAPSS data provides a realistic foundation for failure prediction in industrial settings, as it simulates real-world operational variations, degradation patterns, and failure events. "
            "This approach enables industries such as aerospace, manufacturing, and energy to transition from reactive to predictive maintenance, leading to increased efficiency and cost savings."
        ), width=12)
    ]),
    
    # ======== TRAIN DATASET SECTION ========
    dbc.Row([dbc.Col(html.H2("Train Dataset", className="mb-3"))]),
    dbc.Row([
        dbc.Col(html.P(
            "To develop an effective predictive maintenance model, the C-MAPSS dataset is used, which captures engine run-to-failure data. "
            "This dataset provides sensor readings and operational conditions recorded throughout each engine’s lifecycle, allowing us to predict failures accurately."
        ), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.H4("Dataset Structure", className="mt-3"), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Ul([
            html.Li("Each engine's data is presented as a sequence of operational cycles, capturing the degradation process leading to failure over time."),
            html.Li("Train Dataset: Contains full engine operational histories, where each trajectory ends at failure."),
            html.Li("Remaining Useful Life (RUL) Calculation: RUL = max_cycle - current_cycle, where max_cycle represents the failure point for each engine."),
            html.Li("Testing Data: Contains truncated engine trajectories, meaning the sequences end before failure occurs. The model must predict the remaining useful life (RUL) based on this incomplete data."),
            html.Li("Truth RUL File: A separate file providing the actual RUL values for the engines in the test set, allowing model performance evaluation."),
        ]), width={"size": 8, "offset": 2})
    ]),
    dbc.Row([
        dbc.Col(html.H4("The Operational Settings", className="mt-3"), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Ul([
            html.Li("Operational Setting 1: Altitudes ranging from sea level to 40,000 ft."),
            html.Li("Operational Setting 2: Mach numbers from 0 to 0.90.")
        ]), width={"size": 8, "offset": 2})
    ]),
    dbc.Row([
        dbc.Col(html.H4("The Sensor Readings", className="mt-3"), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Ul([
            html.Li("Sensor 1 (T2): Total temperature at the fan inlet."),
            html.Li("Sensor 2 (P2): Total pressure in the fan."),
            html.Li("Sensor 3 (T24): Total temperature at the Low-Pressure Compressor (LPC) outlet."),
            html.Li("Sensor 4 (T30): Total temperature at the High-Pressure Compressor (HPC) outlet."),
            html.Li("Sensor 5 (T50): Total temperature at the Low-Pressure Turbine (LPT) outlet."),
            html.Li("Sensor 6 (P15): Total pressure in the bypass duct."),
            html.Li("Sensor 7 (P30): Total pressure at the HPC outlet."),
            html.Li("Sensor 8 (Nf): Physical fan speed."),
            html.Li("Sensor 9 (Nc): Physical core speed."),
            html.Li("Sensor 10 (epr): Engine pressure ratio (P50/P2)."),
            html.Li("Sensor 11 (Ps30): Static pressure at the HPC outlet."),
            html.Li("Sensor 12 (phi): Ratio of fuel flow to Ps30."),
            html.Li("Sensor 13 (NRf): Corrected fan speed."),
            html.Li("Sensor 14 (NRc): Corrected core speed."),
            html.Li("Sensor 15 (BPR): Bypass Ratio."),
            html.Li("Sensor 16 (htBleed): Bleed enthalpy."),
            html.Li("Sensor 17 (W31): High-pressure turbine coolant bleed."),
            html.Li("Sensor 18 (W32): Low-pressure turbine coolant bleed."),
            html.Li("Sensors 19-21: Other proprietary or derived parameters related to engine performance."),
        ]), width={"size": 8, "offset": 2})
    ]),
    dbc.Row([
        dbc.Col(html.P(
            "By leveraging these operational settings and sensor readings, machine learning models can identify failure patterns and predict when maintenance is required, "
            "helping businesses reduce downtime and optimize maintenance planning."
        ), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
                id='train-dataset-table',
                columns=[{'name': col, 'id': col} for col in df_train.columns],
                data=df_train.to_dict('records'),  # Show all rows
                style_table={'width': '100%', 'overflowX': 'auto'},  # Full width with horizontal scroll
                style_header={'textAlign': 'center', 'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},  # Center column names
                style_cell={'textAlign': 'center', 'padding': '10px', 'whiteSpace': 'normal'},  # Center column values
                page_size=10,  # Ensures pagination is active
                page_action='native',  # Enables native pagination
                page_current=0,  # Start at the first page
            )
        ], width=12)  # Full width
    ]),
    
     # Add RUL Distribution section
    dbc.Row([dbc.Col(html.H4("RUL Distribution", className="mb-3"))]),
    dbc.Row([dbc.Col(dcc.Graph(id='rul-distribution-plot'), width=12)]),
    dbc.Col(html.P(
    
    ), width=12),
    
    dbc.Col(
    html.Ul([
        html.Li("Majority of engines have RUL below 150 cycles."),
        html.Li("The highest frequency is for engines with RUL between 0-50 cycles."),
        html.Li("The second most common range is RUL between 50-150 cycles."),
        html.Li("Very few engines have high RUL values (200+ cycles)."),
        html.Li("There are significantly fewer engines with RUL above 200 cycles."),
        html.Li("A very small fraction of engines have RUL values above 300 cycles."),
    ]), 
    width=12),

    # ======== FEATURE ENGINEERING SECTION ======== 
    # ======== FEATURE ENGINEERING SECTION ========
    dbc.Row([dbc.Col(html.H2("Feature Engineering", className="mb-3"))]),
    dbc.Col(html.P(
    "To enhance the predictive power of the model and ensure consistency across different sensor readings, two types of normalization techniques were applied."
), width=12),

    dbc.Col(html.P("Min-Max Normalization scales the sensor readings to a fixed range, typically [0,1], ensuring that all features contribute equally to the model without being influenced by differing scales. The transformation formula is:"), width=12),
   
    dbc.Col(dcc.Markdown(r"""
$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$
"""), width=12),


    dbc.Col(html.P("- X_norm: Normalized feature value"), width=12),
    dbc.Col(html.P("- X: Original feature value"), width=12),
    dbc.Col(html.P("- X_min: Minimum value of the feature"), width=12),
    dbc.Col(html.P("- X_max: Maximum value of the feature"), width=12),

    dbc.Col(html.P("This method preserves the relationships between data points while constraining values within a standardized range."), width=12),

    dbc.Col(html.P("In this approach, only the first N values of each engine's operational cycle are used to calculate normalization parameters (mean and standard deviation). The transformation is applied as follows:"), width=12),

    dbc.Col(dcc.Markdown(r"""
$$X_{norm} = \frac{X - \mu_N}{\sigma_N}$$
"""), width=12),

    dbc.Col(html.P("- X_norm: Normalized feature value"), width=12),
    dbc.Col(html.P("- X: Original feature value"), width=12),
    dbc.Col(html.P("- \(\mu_N\): Mean of the first N values of the feature"), width=12),
    dbc.Col(html.P("- \(\sigma_N\): Standard deviation of the first N values"), width=12),

    dbc.Col(html.P("This method ensures that early operational values serve as a reference, making it particularly useful when the dataset represents progressive degradation over time. It helps in capturing deviations from the initial healthy state of the engine."), width=12),

    dbc.Col(html.P("By applying these normalization techniques, the dataset becomes more uniform, improving model convergence and ensuring more reliable failure predictions."), width=12),
    
    dbc.Row([dbc.Col(html.H3("Sensor Time Series Visualization"), className="mb-4")]),

    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='dataset-dropdown',
            options=[{'label': v, 'value': k} for k, v in dataset_options.items()],
            value='df_train_eda',
            placeholder="Select Dataset"
        ), width=4),

        dbc.Col(dcc.Dropdown(
            id='engine-dropdown',
            options=[{'label': f'Engine {i}', 'value': i} for i in engine_ids],
            placeholder="Select Engine",
            value=1
        ), width=4),

        dbc.Col(dcc.Dropdown(
            id='sensor-dropdown_1',
            options=[{'label': col, 'value': col} for col in sensor_columns],
            value='s11',
            placeholder="Select Sensor"
        ), width=4)
    ]),

    dbc.Row([dbc.Col(html.Img(id='sensor-time-series-plot', style={'width': '100%'}))]),

    # ======== EXPLORATORY DATA ANALYSIS (EDA) ========
    dbc.Row([dbc.Col(html.H2("Exploratory Data Analysis", className="mb-3"))]),
    dbc.Row([dbc.Col(html.P(
        "The simulation has been conducted multiple times for the same engine, meaning that each engine undergoes several run-to-failure scenarios. For instance, Engine 1 may have multiple independent degradation cycles recorded, each representing a distinct failure event. Given this variability, it is crucial to analyze the average life cycle of each engine across all simulations. This provides valuable insights into the typical operational lifespan before failure, helping to identify trends in engine degradation and optimize maintenance planning. The table below presents the average life cycle for each engine, offering a comprehensive overview of expected operational duration before failure occurs."
    ), width=12)]),

    # Average Lifecycle Dataset (Under EDA)
    dbc.Row([dbc.Col(html.H3("Average Lifecycle Dataset", className="mb-2"))]),
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
                id='average-lifecycles-table',
                columns=[{'name': col, 'id': col} for col in average_lifecycles.columns],
                data=average_lifecycles.to_dict('records'),
                style_table={'width': '100%', 'overflowX': 'auto'},
                style_header={'textAlign': 'center', 'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_cell={'textAlign': 'center', 'padding': '10px', 'whiteSpace': 'normal'},
                page_action='native',
                page_current=0,
                page_size=10
            )
        ], width=12)
    ]),

    # Sensor Data Visualization (Under EDA)
    dbc.Row([dbc.Col(html.H3("Sensor Data Visualization", className="mb-2"))]),
    dbc.Col(html.P(
        "The plot provides insights into how sensor readings evolve as the engine degrades over its operational life, visualizing the relationship between Remaining Useful Life (RUL) and sensor behavior to help identify degradation patterns. The rolling mean smoothing technique (window size of 10 cycles) is applied to reduce noise and highlight trends in the sensor data. The x-axis represents RUL (remaining cycles before failure), while the y-axis corresponds to the smoothed sensor readings, making it easier to observe gradual changes. As RUL decreases, sensor readings may exhibit increasing deviations, signaling potential faults. This visualization aids in understanding failure precursors, which is crucial for developing predictive maintenance strategies."
    ), width=12),

    # Sensor selection dropdown
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='sensor-dropdown',
            options=[{'label': v, 'value': k} for k, v in sensor_dictionary.items()],
            value='s2'
        ), width=4)
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id='sensor-plot'), width=12)]),

    

    # ======== Modelling SECTION ========
    dbc.Row([dbc.Col(html.H2("Modelling", className="mb-3"))]),
   

    dbc.Row([
        dbc.Col(html.P(
        "The regression modeling process involved testing multiple regressors, including LightGBM, XGBoost, and Linear Regression, "
        "to predict Remaining Useful Life (RUL). A Grid Search optimization was performed for each model, and the optimized versions "
        "consistently outperformed their default counterparts. Among them, the optimized LightGBM regressor yielded the best performance, "
        "leading to four iterations of experimentation."
    ))
]),
    dbc.Row([
        dbc.Col(html.Ul([
        html.Li("Iteration 1: Applied Min-Max normalization to sensor values."),
        html.Li("Iteration 2: Used normalization based on the first n values of each engine's trajectory."),
        html.Li("Iteration 3: Combined Min-Max normalization with RUL clipping, where values exceeding 121 cycles were capped."),
        html.Li("Iteration 4: Applied normalization by first n values with RUL clipping, refining predictive accuracy.")
    ]))
]),
    dbc.Row([
        dbc.Col(html.P(
        "The performance of each iteration was evaluated using MAE, MSE, RMSE, and R² scores, presented in the table and bar graph. "
        "The results demonstrate how feature scaling and data preprocessing impact model accuracy, ultimately enhancing failure prediction."
    ))
]),



    dbc.Row([
        dbc.Col(html.Div([
            html.H3("Iteration 1"),
            html.Ul([
                html.Li("Normalization with Min-Max Scaler"),
                html.Li("Removal of columns Engine ID, Operational Settings, and Stable Sensors"),
                html.Li("Grid Search Optimization of LightGBM model")
            ])
        ])),
        dbc.Col(html.Div([
            html.H3("Iteration 2"),
            html.Ul([
                html.Li("Normalization with First n-Values"),
                html.Li("Removal of columns Engine ID, Operational Settings, and Stable Sensors"),
                html.Li("Grid Search Optimization of LightGBM model")
            ])
        ])),
    ]),

    dbc.Row([
        dbc.Col(html.Div([
            html.H3("Iteration 3"),
            html.Ul([
                html.Li("Normalization with Min-Max Scaler"),
                html.Li("Removal of columns Engine ID, Operational Settings, and Stable Sensors"),
                html.Li("Clipping the RUL values above 121"),
                html.Li("Grid Search Optimization of LightGBM model"),
            ])
        ])),

        dbc.Col(html.Div([
            html.H3("Iteration 4"),
            html.Ul([
                html.Li("Normalization with First n-Values"),
                html.Li("Removal of columns Engine ID, Operational Settings, and Stable Sensors"),
                html.Li("Grid Search Optimization of LightGBM model")
            ])
        ])),
    ]),

    

    dbc.Row([dbc.Col(dash_table.DataTable(
        id='model-performance-table',
        columns=[{'name': col, 'id': col} for col in df_performance.columns],
        data=df_performance.to_dict('records'),
        style_table={'width': '100%', 'overflowX': 'auto'},
        style_header={'textAlign': 'center', 'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
        style_cell={'textAlign': 'center', 'padding': '10px', 'whiteSpace': 'normal'},
        page_size=10,
    ))]),

    # Bar Plots for Model Performance
    dbc.Row([
        dbc.Col(dcc.Graph(id='mae-bar-plot'), width=6),
        dbc.Col(dcc.Graph(id='mse-bar-plot'), width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='rmse-bar-plot'), width=6),
        dbc.Col(dcc.Graph(id='r2-bar-plot'), width=6),
    ]),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='bias-bar-plot'), width=6),
    ]),

    # ======== CONCLUSION SECTION ========
    dbc.Row([
        dbc.Col(html.H2("Conclusion"))
    ]),
    
    dbc.Row([
        dbc.Col(html.H3("Business Impact of the Predictive Maintenance Dashboard")),
        dbc.Col(html.P("""
            This Predictive Maintenance (PdM) Dashboard leverages machine learning to forecast Remaining Useful Life (RUL) and detect potential failures before they occur. By analyzing sensor readings, operational settings, and degradation patterns, the system enables businesses to reduce unplanned downtime, optimize maintenance schedules, and lower operational costs. With predictive insights, companies can move from a reactive maintenance strategy to a proactive one, leading to increased equipment reliability, extended asset lifespan, and improved safety compliance.
        """)),
    ]),
    dbc.Row([
        dbc.Col(html.H3("Challenges & Limitations")),
        dbc.Col(html.P("""
            While the optimized LightGBM regressor achieves around 90% accuracy, there are still uncertainties and prediction errors that can impact decision-making. Complex real-world conditions, sensor noise, and unseen failure modes may affect the model’s reliability. Additionally, predicting failures far in advance remains difficult, especially when degradation patterns vary significantly across engines.
        """)),
    ]),
    dbc.Row([
        dbc.Col(html.H3("Future Improvements & Classification-Based Predictive Approach")),
        dbc.Col(html.P("""
            To further enhance prediction accuracy, a classification-based approach could be integrated alongside regression models. Instead of predicting exact RUL values, engines could be categorized into risk zones (e.g., "Safe," "Warning," and "Critical") based on a threshold, such as 30 cycles before failure. This approach would simplify decision-making, making it easier for businesses to plan preventive actions at critical moments, ultimately improving maintenance efficiency and failure prevention strategies.
        """)),
    ]),
])

#Callbacks

# Callback to update the RUL distribution plot
@app.callback(
    Output('rul-distribution-plot', 'figure'),
    Input('train-dataset-table', 'data')
)
def update_rul_distribution_plot(data):
    return generate_rul_distribution_plot()

# Callback to update the sensor plot
@app.callback(
    Output('sensor-plot', 'figure'),
    [Input('sensor-dropdown', 'value')]
)
def update_sensor_plot(selected_sensor):
    return generate_sensor_plot(selected_sensor)

# Callback to update the plot
@app.callback(
    Output('sensor-time-series-plot', 'src'),
    [Input('dataset-dropdown', 'value'),
     Input('engine-dropdown', 'value'),
     Input('sensor-dropdown_1', 'value')]
)
def update_time_series_plot(dataset_name, engine_id, sensor_name):
    return generate_time_series_plot(dataset_name, engine_id, sensor_name)


# Callbacks to update bar graphs
@app.callback(
    [Output('mae-bar-plot', 'figure'),
     Output('mse-bar-plot', 'figure'),
     Output('rmse-bar-plot', 'figure'),
     Output('r2-bar-plot', 'figure'),
     Output('bias-bar-plot', 'figure')],
    Input('model-performance-table', 'data')
)
def update_graphs(data):
    df = pd.DataFrame(data)

    mae_fig = go.Figure(data=[go.Bar(x=df['Iteration'], y=df['MAE'], marker_color='blue')])
    mae_fig.update_layout(title="Mean Absolute Error (MAE)", xaxis_title="Iteration", yaxis_title="MAE")

    mse_fig = go.Figure(data=[go.Bar(x=df['Iteration'], y=df['MSE'], marker_color='red')])
    mse_fig.update_layout(title="Mean Squared Error (MSE)", xaxis_title="Iteration", yaxis_title="MSE")

    rmse_fig = go.Figure(data=[go.Bar(x=df['Iteration'], y=df['RMSE'], marker_color='green')])
    rmse_fig.update_layout(title="Root Mean Squared Error (RMSE)", xaxis_title="Iteration", yaxis_title="RMSE")

    r2_fig = go.Figure(data=[go.Bar(x=df['Iteration'], y=df['R²_Score'], marker_color='purple')])
    r2_fig.update_layout(title="R² Score", xaxis_title="Iteration", yaxis_title="R² Score")

    bias_fig = generate_bias_plot()
    

    return mae_fig, mse_fig, rmse_fig, r2_fig, bias_fig


# Run the app
if __name__ == '__main__':
    print("App is running at http://127.0.0.1:8050/")
    app.run_server(debug=True, port=8050)














