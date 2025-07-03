
import streamlit as st
import pandas as pd
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
import numpy as np
import base64

st.set_page_config(page_title="FleetGenie Pro", layout="wide", page_icon="🚚")

# Logo header
def load_logo():
    with open("fleetgenie_logo.png", "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

st.markdown(
    f"<div style='text-align:center'><img src='data:image/png;base64,{load_logo()}' width='120'><h2 style='color:#01796f'>FleetGenie Predictive Maintenance Analytics</h2><hr></div>",
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    return pd.read_csv("synthetic_fleet_survey_advanced.csv")

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
sizes = st.sidebar.multiselect("Company Size", df["Company_Size"].unique(), df["Company_Size"].unique())
strategies = st.sidebar.multiselect("Maintenance Strategy", df["Maint_Strategy"].unique(), df["Maint_Strategy"].unique())
ev_min = st.sidebar.slider("Minimum EV %", 0, 100, 0)

filtered = df[
    df["Company_Size"].isin(sizes) &
    df["Maint_Strategy"].isin(strategies) &
    (df["EVs_%"] >= ev_min)
]

tabs = st.tabs(["📊 Visualisation", "🤖 Classification", "🧩 Clustering", "🔗 Assoc. Rules"])

# --- Visualisation Tab ---
with tabs[0]:
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Fleets", f"{filtered.shape[0]}")
    col2.metric("Avg EV %", f"{filtered['EVs_%'].mean():.1f}")
    col3.metric("Dominant Strategy", filtered['Maint_Strategy'].mode()[0])

    st.markdown("### Company Size Distribution")
    st.plotly_chart(px.histogram(filtered, x="Company_Size", color="Company_Size"), use_container_width=True)

    st.markdown("### EV % by Company Size")
    st.plotly_chart(px.box(filtered, x="Company_Size", y="EVs_%", color="Company_Size"), use_container_width=True)

# --- Classification Tab ---
with tabs[1]:
    st.subheader("Predict AI Adoption (Using / Highly Interested)")
    data = filtered.copy()
    data["Target"] = data["AI_Adoption"].isin(["Using","Highly Interested"]).astype(int)
    X = data.drop(columns=["Target"])
    y = data["Target"]

    cat = X.select_dtypes(include="object").columns.tolist()
    num = X.select_dtypes(exclude="object").columns.tolist()
    pre = ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat), ("num",StandardScaler(),num)])

    models = {
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "GradientBoost": GradientBoostingClassifier()
    }
    results = []
    roc_info = []
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
    for name,model in models.items():
        pipe = Pipeline([("prep",pre),("model",model)])
        pipe.fit(X_train,y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:,1]
        results.append([name,
                        accuracy_score(y_test,y_pred),
                        precision_score(y_test,y_pred),
                        recall_score(y_test,y_pred),
                        f1_score(y_test,y_pred)])
        fpr,tpr,_ = roc_curve(y_test,y_prob)
        roc_info.append((name,fpr,tpr,auc(fpr,tpr)))

    st.dataframe(pd.DataFrame(results,columns=["Model","Acc","Prec","Rec","F1"]).set_index("Model").style.format("{:.2f}"))

    st.markdown("#### ROC Curves")
    import plotly.graph_objects as go
    roc_fig = go.Figure()
    for name,fpr,tpr,auc_val in roc_info:
        roc_fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',name=f"{name} (AUC {auc_val:.2f})"))
    roc_fig.add_shape(type='line',x0=0,y0=0,x1=1,y1=1,line=dict(dash='dash'))
    roc_fig.update_layout(xaxis_title="FPR",yaxis_title="TPR")
    st.plotly_chart(roc_fig,use_container_width=True)

# --- Clustering Tab ---
with tabs[2]:
    st.subheader("K-Means Clustering on EV% and Trucks%")
    X_clu = filtered[["EVs_%","Trucks_%"]]
    k = st.slider("Number of clusters",2,10,3)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_clu)
    filtered["Cluster"] = kmeans.labels_
    st.plotly_chart(px.scatter(filtered,x="EVs_%",y="Trucks_%",color="Cluster",hover_data=["Company_Size","Maint_Strategy"]), use_container_width=True)
    st.write("Cluster Centers:", kmeans.cluster_centers_)

# --- Association Rules Tab ---
with tabs[3]:
    st.subheader("Association Rules on Key Benefits")
    transactions = filtered["Key_Benefits"].str.split(", ")
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary,columns=te.columns_)
    minsup = st.slider("Min Support",0.01,1.0,0.1,0.01)
    minconf = st.slider("Min Confidence",0.1,1.0,0.6,0.05)
    freq = apriori(df_trans,min_support=minsup,use_colnames=True)
    rules = association_rules(freq,metric="confidence",min_threshold=minconf).sort_values("confidence",ascending=False).head(10)
    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])
