"""AQTIS Dashboard Theme — dark trading theme for Plotly + Streamlit."""

AQTIS_COLORS = {
    "background": "#0e1117",
    "card_bg": "#1a1d24",
    "text": "#fafafa",
    "green": "#00d26a",
    "red": "#ff6b6b",
    "blue": "#4dabf7",
    "yellow": "#ffd43b",
    "purple": "#da77f2",
    "muted": "#636e72",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=AQTIS_COLORS["background"],
    plot_bgcolor=AQTIS_COLORS["card_bg"],
    font=dict(color=AQTIS_COLORS["text"], family="monospace"),
    xaxis=dict(gridcolor="#2d3436", zeroline=False),
    yaxis=dict(gridcolor="#2d3436", zeroline=False),
    margin=dict(l=50, r=20, t=40, b=40),
    hovermode="x unified",
)


def apply_theme(fig):
    """Apply AQTIS dark theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig
