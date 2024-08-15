import plotly.express as px
import plotly.graph_objects as go

def display_img(img, label=False):
    fig = px.imshow(img, color_continuous_scale='gray')
    fig.update_layout(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    if label:
        label = chr(label + ord('A'))
        fig.update_layout(
            title=f'This is a {label.upper()}'
        )
    fig.show()
    
def display_metrics(costs= False, train_accs= False, test_accs= False, title="Evolution of Cost, and Training/Test Accuracy over the Training Steps"):
    fig=go.Figure()
    if costs: 
        fig.add_trace(go.Scatter(
            x=list(range(len(costs))),
            y=costs,
            mode='lines',
            name='Cost'
        ))

    if train_accs:
        fig.add_trace(go.Scatter(
            x=list(range(len(train_accs))),
            y=train_accs,
            mode='lines',
            name='Train Accuracy'
        ))
    
    if test_accs:
        fig.add_trace(go.Scatter(
            x=list(range(len(test_accs))),
            y=test_accs,
            mode='lines',
            name='Test Accuracy'
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Epochs"
        ),
        yaxis=dict(
            title="Value",
            range=[0,1]
        ),
        legend_title="Metrics"
    )
    fig.show()
def compare_metric(metrics, title, hyperparameters):
    fig=go.Figure()
    for i in range (0, len(metrics)):
        metric = metrics[i]
        hyperparameter = hyperparameters[i]
        customdata = [[hyperparameter['learning_rate'], hyperparameter['hidden_nodes'], 
                       hyperparameter['minibatch_size'], hyperparameter['l2_lambda']]] * len(metric)
        fig.add_trace(go.Scatter(
            x=[i * 10 for i in range(len(metric))],
            y=metric,
            mode='lines',
            name=f'Metric {i}',
            customdata=customdata,
            hovertemplate=('<b> Epoch: %{x}</b> <br></br>'+
                           '<b> Value: %{y}</b> <br></br>'+
                           'Learning rate: %{customdata[0]} <br></br>'+
                           'Hidden nodes: %{customdata[1]} <br></br>'+
                           'Mini batch size: %{customdata[2]} <br></br>'+
                           'L2 lambda: %{customdata[3]} <br></br>'+
                           '<extra></extra>')
        ))
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Epochs"
        ),
        yaxis=dict(
            title="Value",
            range=[0,1]
        ),
        legend_title="Metrics"
    )
    fig.show()
    return fig
    
def save_plots_html(filename, train_plot, test_plot):
    path=f"hypertuning/plots/train_acc/{filename}.html"
    train_plot.write_html(path)
    
    path=f"hypertuning/plots/test_acc/{filename}.html"
    test_plot.write_html(path)
    
    print("Html plots saved")