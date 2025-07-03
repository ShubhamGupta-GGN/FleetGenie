
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.graph_objects as go

st.set_page_config(page_title="FleetGenie Dashboard", layout="wide", page_icon="🚚")

st.markdown("""
<style>
.block-container { padding-top: 1rem; }
.card { background: #fff; padding: 1.2rem 1.5rem; border-radius: 1.1rem; box-shadow: 0 2px 8px rgba(0,0,0,.06); margin-bottom: 28px }
.kpi-title { font-size:1.08rem;color:#555 }
.kpi-value { font-size:2.1rem;font-weight:700;color:#16796f }
hr { margin-top: .2rem; margin-bottom: 1.2rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("synthetic_fleet_survey_advanced.csv")
df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filter Data")
siz = st.sidebar.multiselect("Company Size", df["Company_Size"].unique(), df["Company_Size"].unique())
strat = st.sidebar.multiselect("Maintenance Strategy", df["Maint_Strategy"].unique(), df["Maint_Strategy"].unique())
ev_min = st.sidebar.slider("Minimum EV (%)", 0, 100, 0)
df = df[df["Company_Size"].isin(siz) & df["Maint_Strategy"].isin(strat) & (df["EVs_%"] >= ev_min)]

# KPI
st.markdown("<h1 style='color:#004b6b'>FleetGenie Dashboard</h1><hr>", unsafe_allow_html=True)
k1,k2,k3,k4 = st.columns(4)
k1.markdown(f"<div class='card'><div class='kpi-title'>Fleets</div><div class='kpi-value'>{df.shape[0]}</div></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='card'><div class='kpi-title'>Avg EV%</div><div class='kpi-value'>{df['EVs_%'].mean():.1f}</div></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='card'><div class='kpi-title'>Avg Trucks%</div><div class='kpi-value'>{df['Trucks_%'].mean():.1f}</div></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='card'><div class='kpi-title'>Top Strategy</div><div class='kpi-value'>{df['Maint_Strategy'].mode()[0]}</div></div>", unsafe_allow_html=True)

tab1,tab2,tab3,tab4,tab5 = st.tabs(["📊 Visuals","🤖 Classification","📈 Regression","🧩 Clustering","🔗 Association"])

with tab1:
    st.subheader("Visual Insights")
    c1,c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(df, x="Company_Size", color="Company_Size", template="plotly_white"), use_container_width=True)
        st.plotly_chart(px.histogram(df, x="Maint_Strategy", color="Maint_Strategy", template="plotly_white"), use_container_width=True)
    with c2:
        st.plotly_chart(px.box(df, x="Company_Size", y="EVs_%", color="Company_Size", template="plotly_white"), use_container_width=True)
        st.plotly_chart(px.histogram(df, x="Breakdowns_Year", color="Company_Size", template="plotly_white"), use_container_width=True)

with tab2:
    st.subheader("AI Adoption Classification")
    df["Target"] = df["AI_Adoption"].isin(["Using","Highly Interested"]).astype(int)
    X = df.drop(columns=["Target"])
    y = df["Target"]
    cat = X.select_dtypes("object").columns
    num = X.select_dtypes(exclude="object").columns
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("num", StandardScaler(), num)])
    models = {"KNN":KNeighborsClassifier(), "DT":DecisionTreeClassifier(), "RF":RandomForestClassifier(), "GB":GradientBoostingClassifier()}
    res, roc_data = [], []
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    for n,m in models.items():
        pipe = Pipeline([("prep", pre), ("model", m)])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_ts)
        y_prob = pipe.predict_proba(X_ts)[:,1]
        res.append([n, accuracy_score(y_ts,y_pred), precision_score(y_ts,y_pred), recall_score(y_ts,y_pred), f1_score(y_ts,y_pred)])
        fpr,tpr,_ = roc_curve(y_ts,y_prob)
        roc_data.append((n,fpr,tpr,auc(fpr,tpr)))
    st.dataframe(pd.DataFrame(res, columns=["Model","Acc","Prec","Rec","F1"]).set_index("Model").style.format("{:.2f}"))
    fig = go.Figure()
    for n,fpr,tpr,aucv in roc_data:
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',name=f"{n} AUC={aucv:.2f}"))
    fig.add_shape(type='line',x0=0,y0=0,x1=1,y1=1,line=dict(dash='dash'))
    fig.update_layout(template='plotly_white', title='ROC Curves', xaxis_title='FPR', yaxis_title='TPR')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Downtime Regression")
    map_dt = {"<1":0.5,"1-5":3,"6-10":8,"11+":12}
    df["Downtime_Num"] = df["Downtime_%"].map(map_dt)
    Xr = df.drop(columns=["Downtime_Num"])
    yr = df["Downtime_Num"]
    cat_r = Xr.select_dtypes("object").columns
    num_r = Xr.select_dtypes(exclude="object").columns
    pre_r = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_r), ("num", StandardScaler(), num_r)])
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.tree import DecisionTreeRegressor
    regs = {"Linear":LinearRegression(), "Ridge":Ridge(), "Lasso":Lasso(alpha=0.1), "DT":DecisionTreeRegressor()}
    scores=[]
    for n,m in regs.items():
        pipe = Pipeline([("prep", pre_r), ("model", m)])
        pipe.fit(Xr, yr)
        scores.append([n, pipe.score(Xr, yr)])
    st.dataframe(pd.DataFrame(scores, columns=["Model","R²"]).set_index("Model").style.format("{:.2f}"))

with tab4:
    st.subheader("K-Means Clustering (EV% vs Trucks%)")
    X_clu = df[["EVs_%","Trucks_%"]]
    k = st.slider("Clusters", 2, 10, 4)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_clu)
    df["Cluster"] = kmeans.labels_
    st.plotly_chart(px.scatter(df, x="EVs_%", y="Trucks_%", color="Cluster", hover_data=["Company_Size","Maint_Strategy"], template="plotly_white"), use_container_width=True)
    st.write("Cluster Centers:", kmeans.cluster_centers_)

with tab5:
    st.subheader("Association Rules (Key_Benefits)")
    transactions = df["Key_Benefits"].str.split(", ")
    te=TransactionEncoder()
    ohe = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(ohe, columns=te.columns_)
    sup = st.slider("Min Support", 0.01, 1.0, 0.1, 0.01)
    conf = st.slider("Min Confidence", 0.1, 1.0, 0.6, 0.05)
    freq = apriori(df_trans, min_support=sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=conf).sort_values("confidence", ascending=False).head(10)
    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])
