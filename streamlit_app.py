import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.graph_objects as go

st.set_page_config(page_title="FleetGenie Dashboard", layout="wide", page_icon="🚚")

# ---- Custom CSS for KPI Cards and Visuals ----
st.markdown("""
<style>
.block-container { padding-top: 1rem; }
.card { background: #fff; padding: 1.2rem 1.5rem; border-radius: 1.1rem; box-shadow: 0 2px 8px rgba(0,0,0,.06); margin-bottom: 28px }
.kpi-title { font-size:1.08rem;color:#555 }
.kpi-value { font-size:2.1rem;font-weight:700;color:#16796f }
hr { margin-top: .2rem; margin-bottom: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ---- Load Data ----
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_fleet_survey_advanced.csv")
df = load_data()

# ---- Sidebar Filters ----
st.sidebar.header("🔍 Filter Data")
with st.sidebar:
    company = st.multiselect("Company Size", df["Company_Size"].unique(), df["Company_Size"].unique())
    maint = st.multiselect("Maintenance Strategy", df["Maint_Strategy"].unique(), df["Maint_Strategy"].unique())
    ev_min = st.slider("Minimum EV (%)", 0, 100, 0)
    df = df[df["Company_Size"].isin(company) & df["Maint_Strategy"].isin(maint) & (df["EVs_%"] >= ev_min)]

# ---- KPI Cards ----
st.markdown("<h1 style='color:#004b6b'>FleetGenie: Predictive Maintenance Dashboard</h1><hr>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1: st.markdown(f"<div class='card'><div class='kpi-title'>Fleets</div><div class='kpi-value'>{df.shape[0]}</div></div>", unsafe_allow_html=True)
with col2: st.markdown(f"<div class='card'><div class='kpi-title'>Avg EV%</div><div class='kpi-value'>{df['EVs_%'].mean():.1f}</div></div>", unsafe_allow_html=True)
with col3: st.markdown(f"<div class='card'><div class='kpi-title'>Avg Trucks%</div><div class='kpi-value'>{df['Trucks_%'].mean():.1f}</div></div>", unsafe_allow_html=True)
with col4: st.markdown(f"<div class='card'><div class='kpi-title'>Top Strategy</div><div class='kpi-value'>{df['Maint_Strategy'].mode()[0]}</div></div>", unsafe_allow_html=True)

# ---- Page Sections with Tabs ----
tabs = st.tabs([
    "📊 Visual Insights",
    "🤖 Classification",
    "🧩 Clustering",
    "🔗 Association Rules",
    "📈 Regression"
])

# ---- VISUAL INSIGHTS TAB ----
with tabs[0]:
    st.subheader("Business Insights")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.histogram(df, x="Company_Size", color="Company_Size", template="plotly_white", title="Company Size Distribution")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.box(df, x="Company_Size", y="EVs_%", color="Company_Size", template="plotly_white", title="EV Share by Company Size")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.histogram(df, x="Maint_Strategy", color="Maint_Strategy", template="plotly_white", title="Maintenance Strategy Split")
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        fig4 = px.histogram(df, x="Breakdowns_Year", color="Company_Size", template="plotly_white", title="Breakdown Frequency")
        st.plotly_chart(fig4, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- CLASSIFICATION TAB ----
with tabs[1]:
    st.subheader("AI Adoption Classification")
    df["Target"] = df["AI_Adoption"].isin(["Using","Highly Interested"]).astype(int)
    X = df.drop(columns=["Target"])
    y = df["Target"]
    cat = X.select_dtypes("object").columns
    num = X.select_dtypes(exclude="object").columns
    pre = ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat), ("num",StandardScaler(),num)])
    models = {"KNN":KNeighborsClassifier(),
              "DT":DecisionTreeClassifier(),
              "RF":RandomForestClassifier(),
              "GB":GradientBoostingClassifier()}
    res, roc_data = [], []
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    for n, m in models.items():
        pipe = Pipeline([("prep", pre), ("model", m)])
        pipe.fit(X_tr, y_tr)
        y_pr = pipe.predict(X_ts)
        y_proba = pipe.predict_proba(X_ts)[:,1]
        res.append([n, accuracy_score(y_ts,y_pr), precision_score(y_ts,y_pr),
                    recall_score(y_ts,y_pr), f1_score(y_ts,y_pr)])
        fpr, tpr, _ = roc_curve(y_ts, y_proba)
        roc_data.append((n, fpr, tpr, auc(fpr, tpr)))
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Model Performance")
    st.dataframe(pd.DataFrame(res, columns=["Model","Acc","Prec","Rec","F1"]).set_index("Model").style.format("{:.2f}"))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### ROC Curves")
    fig = go.Figure()
    for n, fpr, tpr, aucv in roc_data:
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{n} (AUC {aucv:.2f})"))
    fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash'))
    fig.update_layout(xaxis_title="FPR", yaxis_title="TPR", template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- CLUSTERING TAB ----
with tabs[2]:
    st.subheader("Fleet Segmentation (K‑Means Clustering)")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    X_clu = df[["EVs_%","Trucks_%"]]
    k = st.slider("Number of clusters", 2, 10, 4)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_clu)
    df["Cluster"] = kmeans.labels_
    st.plotly_chart(px.scatter(df, x="EVs_%", y="Trucks_%", color="Cluster", hover_data=["Company_Size","Maint_Strategy"], template="plotly_white", title="Cluster Segmentation"), use_container_width=True)
    st.write("Cluster Centers:", kmeans.cluster_centers_)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- ASSOCIATION RULES TAB ----
with tabs[3]:
    st.subheader("Association Rule Mining (Key Benefits)")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    trans = df["Key_Benefits"].str.split(", ")
    te = TransactionEncoder()
    onehot = te.fit(trans).transform(trans)
    df_one = pd.DataFrame(onehot, columns=te.columns_)
    sup = st.slider("Min Support", 0.01, 1.0, 0.1, 0.01)
    conf = st.slider("Min Confidence", 0.1, 1.0, 0.6, 0.05)
    freq = apriori(df_one, min_support=sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=conf).sort_values("confidence", ascending=False).head(10)
    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])
    st.markdown("</div>", unsafe_allow_html=True)

# ---- REGRESSION TAB ----
with tabs[4]:
    st.subheader("Regression Insights (Downtime Prediction)")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    dt_map = {"<1":0.5, "1-5":3, "6-10":8, "11+":12}
    df["Downtime_Num"] = df["Breakdowns_Year"].map(dt_map).fillna(3)
    Xr = df.drop(columns=["Downtime_Num"])
    yr = df["Downtime_Num"]
    catr = Xr.select_dtypes("object").columns
    numr = Xr.select_dtypes(exclude="object").columns
    prer = ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),catr), ("num",StandardScaler(),numr)])
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.tree import DecisionTreeRegressor
    regs = {"Linear":LinearRegression(),"Ridge":Ridge(alpha=1.0),"Lasso":Lasso(alpha=0.1),"DT Regr":DecisionTreeRegressor()}
    r2s = []
    for n, m in regs.items():
        pipe = Pipeline([("prep",prer),("model",m)])
        pipe.fit(Xr,yr)
        pred = pipe.predict(Xr)
        r2s.append([n, pipe.score(Xr,yr)])
        fig_pred = px.scatter(x=yr, y=pred, labels={'x':"Actual Downtime", 'y':"Predicted"}, title=f"{n}: Predicted vs Actual")
        st.plotly_chart(fig_pred, use_container_width=True)
    st.markdown("### Model R² Scores")
    st.dataframe(pd.DataFrame(r2s, columns=["Model", "R²"]).set_index("Model").style.format("{:.2f}"))
    st.markdown("</div>", unsafe_allow_html=True)
