import plotly.express as px

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