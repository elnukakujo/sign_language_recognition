import plotly.express as px
import plotly.graph_objects as go

def display_img(img, label=False):
    fig = px.imshow(img)
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
    
def display_metrics(costs= False, train_accs= False, test_accs= False):
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
        title="Evolution of Cost, and Training/Test Accuracy over the Training Steps",
        xaxis=dict(
            title="Epochs"
        ),
        yaxis=dict(
            title="Value",
            range=[0,1.5]
        ),
        legend_title="Metrics"
    )
    fig.show()