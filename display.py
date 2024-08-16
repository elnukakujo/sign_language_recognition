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
    
def display_metrics(metrics, title, hyperparameters):
    fig=go.Figure()
    
    if type(hyperparameters['lr'])==list:
        customdata = []
        for i in range(len(metrics["cost"])):
            hyperparameter = [hyperparameters['lr'][i], 
                            hyperparameters['hidden_nodes'][i], 
                            hyperparameters['minibatch_size'][i], 
                            hyperparameters['l2_lambda'][i]]
            for _ in range(len(metrics["cost"][i])):
                customdata.append(hyperparameter.copy())
    else:
        customdata=[[hyperparameters['lr'], 
                    hyperparameters['hidden_nodes'], 
                    hyperparameters['minibatch_size'], 
                    hyperparameters['l2_lambda']]]*len(metrics["cost"][0])
            
    flattened_metrics = dict()
    for key, value in metrics.items():
        flattened_metric = []
        for sublist in value:
            flattened_metric.extend(sublist)  # Add all elements from the sublist to the flattened list
        flattened_metrics[key] = flattened_metric
    
    for key, values in flattened_metrics.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(values))),
            y=values,
            mode='lines',
            name=key,
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

def compare_metric(metrics, title, hyperparameters):
    fig=go.Figure()
    for i in range (0, len(metrics)):
        metric = metrics[i]
        hyperparameter = hyperparameters[i]
        customdata = [[hyperparameter['lr'], hyperparameter['hidden_nodes'], 
                       hyperparameter['minibatch_size'], hyperparameter['l2_lambda']]] * len(metric)
        fig.add_trace(go.Scatter(
            x=[i for i in range(len(metric))],
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
    
def save_plot_html(path,plot):
    plot.write_html(path)
    print(f"Plot html saved at {path}")

def save_plots_html(path, filename, train_plot, test_plot):
    path=f"{path}plots/train_acc/{filename}.html"
    train_plot.write_html(path)
    
    path=f"{path}plots/test_acc/{filename}.html"
    save_plot_html(path,test_plot)
    
    print("Html plots saved")