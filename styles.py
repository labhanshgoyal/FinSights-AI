import streamlit as st

def apply_theme():
    """Inject premium dark-mode CSS theme into the Streamlit app."""
    st.markdown("""
    <style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Hide Streamlit defaults ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Main background ── */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0f1a 100%);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #111827 100%) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        background: linear-gradient(135deg, #818cf8, #6366f1, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ── Glassmorphism Cards ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.1);
        transform: translateY(-2px);
    }

    /* ── Stock Header ── */
    .stock-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.05));
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .stock-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .stock-ticker {
        font-size: 0.8rem;
        color: #818cf8;
        letter-spacing: 3px;
        text-transform: uppercase;
        font-weight: 600;
    }
    .stock-name {
        font-size: 2rem;
        font-weight: 800;
        color: #f1f5f9;
        margin: 0.2rem 0;
        letter-spacing: -0.5px;
    }
    .stock-price {
        font-size: 2.5rem;
        font-weight: 700;
        color: #fff;
        margin-right: 1rem;
    }
    .stock-change-up {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
    }
    .stock-change-down {
        display: inline-block;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #f87171;
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
    }

    /* ── Live pulse dot ── */
    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #34d399;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.5); }
        70% { box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }
        100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 0.3rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        color: #94a3b8;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #fff !important;
        font-weight: 600;
        border-radius: 10px;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-2px);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 500;
    }
    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-weight: 700;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }

    /* ── Chat Panel ── */
    .chat-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.06));
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px 16px 0 0;
        padding: 1rem 1.2rem;
        margin-bottom: 0;
    }
    .chat-header-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .chat-header-sub {
        font-size: 0.75rem;
        color: #818cf8;
        margin-top: 0.15rem;
    }
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        margin-bottom: 0.5rem;
        padding: 0.8rem;
    }
    /* Style the scrollable chat container */
    .chat-scroll-area [data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }

    /* ── Plotly chart container ── */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Section dividers ── */
    hr {
        border-color: rgba(99, 102, 241, 0.15) !important;
    }

    /* ── AI Analysis box ── */
    .ai-analysis-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.06), rgba(139, 92, 246, 0.04));
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.3); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.5); }
    </style>
    """, unsafe_allow_html=True)


def render_stock_header(ticker, stock_name, current_price, price_change, pct_change):
    """Render the premium stock header card with animated pulse dot and change badge."""
    arrow = "▲" if price_change >= 0 else "▼"
    change_class = "stock-change-up" if price_change >= 0 else "stock-change-down"

    st.markdown(f"""
    <div class="stock-header">
        <div class="stock-ticker"><span class="pulse-dot"></span> {ticker}</div>
        <div class="stock-name">{stock_name}</div>
        <div style="display: flex; align-items: center; gap: 1rem; margin-top: 0.5rem;">
            <span class="stock-price">₹{current_price:.2f}</span>
            <span class="{change_class}">
                {arrow} {abs(price_change):.2f} ({abs(pct_change):.2f}%)
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_ai_analysis(content):
    """Render AI analysis inside a styled glass card."""
    st.markdown(f"""
    <div class="ai-analysis-box">
        {content}
    </div>
    """, unsafe_allow_html=True)
