import dash
import pandas as pd
import numpy as np
# Dash and Plotly
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
# Sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Backend to generate dataset
def gen_dat(n_samples,noise,dataset_type):
    if dataset_type == 'non_lin1':
        return datasets.make_circles(n_samples=n_samples,noise=noise,factor=0.5,random_state=1)
    elif dataset_type == 'non_lin2':
        return datasets.make_moons(n_samples=n_samples,noise=noise,random_state=0)
    elif dataset_type == 'lin':
        X,Y = datasets.make_classification(n_samples=n_samples,n_features=2,n_redundant=0,n_informative=2,random_state=2,n_clusters_per_class=1)
        # Add noise to features
        range_noise = np.random.RandomState(seed=2)
        X += noise*(range_noise.uniform(size=X.shape))
        lin_data = (X,Y)
        return lin_data
    else:
        raise ValueError("Select correct dataset type !")
       
# Initializing instance
app = dash.Dash(__name__)
server = app.server

# Laying out the frontend
app.layout = html.Div(children=[
    html.Div(className="banner",children=[
        # Set name
        html.Div(className="container scalable",children=[
            html.H1('Support Vector Machine(SVM) Visual Tool')
        ]),
    ]),
])


# Run
if __name__ == "__main__":
    app.run_server(debug=True)
