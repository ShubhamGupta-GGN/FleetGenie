import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd, plotly.express as px, numpy as np
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
import base64

st.set_page_config(page_title="FleetGenie Dashboard", layout="wide", page_icon="🚚")

# --------- CSS: Card style, Font, Colors ----------
st.markdown("""
<style>
/* Hero header and card styling */
.block-container { padding-top: 0.5rem; }
.card { background: #fff; padding: 1.1rem 1.5rem; border-radius: 1.1rem; box-shadow: 0 2px 8px rgba(0,0,0,.06); margin-bottom: 28px }
h1, h2, h3 { color: #004b6b !important; }
hr { margin-top: .3rem; margin-bottom: 1.1rem; }
.metric-label { font-size:1.02rem;color:#888 }
.metric-value { font-size:2rem;font-weight:600;color:#01796f }
</style>
""", unsafe_allow_html=True)

# --------- Header & Logo ----------
def logo_base64():
    with open("fleetgenie_logo.png","rb") as img:
        return base64.b64encode(img.read()).decode()
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:1rem">
        <img src='data:image/png;base64,{logo_base64()}' width='75'/>
        <h1 style="margin-bottom:0;">FleetGenie Dashboard</h1>
    </div><hr style='margin-top:4px;margin-bottom:20px'>
    """,
    unsafe_allow_html=True
)

# --------- Data ---------
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_fleet_survey_advanced.csv")
df = load_data()

# --------- Sidebar Filters in Expander ---------
with st.sidebar:
    st.image("fleetgenie_logo.png", width=120)
    with st.expander("Filters", expanded=True):
        size_sel = st.multiselect("Company Size", df["Company_Size"].unique(), df["Company_Size"].unique())
        strat_sel = st.multiselect("Maintenance Strategy", df["Maint_Strategy"].unique(), df["Maint_Strategy"].unique())
        ev_min = st.slider("Min EV %", 0, 100, 0)
    df = df[df["Company_Size"].isin(size_sel) & df["Maint_Strategy"].isin(strat_sel) & (df["EVs_%"]>=ev_min)]

# --------- Option Menu Navigation ---------
selected = option_menu(
    menu_title=None,
    options=["Visualisation","Classification","Clustering","Association"],
    icons=["bar-chart","cpu","diagram-3","link-45deg"],
    orientation="horizontal",
    styles={
        "container":{"padding":"0!important","background-color":"#f0f4f7"},
        "icon":{"font-size":"19px"},
        "nav-link":{"font-size":"16px","margin":"0px","--hover-color":"#e4eef2"},
        "nav-link-selected":{"background-color":"#004b6b","color":"#fff"},
    }
)

# --------- TAB 1: Visualisation ---------
if selected=="Visualisation":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1,col2,col3 = st.columns(3)
    col1.metric("Fleets", df.shape[0])
    col2.metric("Avg EV%", f"{df['EVs_%'].mean():.1f}")
    col3.metric("Top Strategy", df["Maint_Strategy"].mode()[0])
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### EV Share by Company Size")
    st.plotly_chart(px.box(df, x="Company_Size", y="EVs_%", color="Company_Size", template="plotly_white"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Maintenance Strategy Split")
    st.plotly_chart(px.histogram(df, x="Maint_Strategy", color="Maint_Strategy", template="plotly_white"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Breakdown Frequency")
    st.plotly_chart(px.histogram(df, x="Breakdowns_Year", color="Company_Size", template="plotly_white"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --------- TAB 2: Classification ---------
elif selected=="Classification":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Classify AI Adoption")
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
    res=[]
    roc_data=[]
    X_tr,X_ts,y_tr,y_ts = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
    for n,m in models.items():
        pipe = Pipeline([("prep",pre),("model",m)])
        pipe.fit(X_tr,y_tr)
        y_pr=pipe.predict(X_ts)
        y_proba=pipe.predict_proba(X_ts)[:,1]
        res.append([n, accuracy_score(y_ts,y_pr), precision_score(y_ts,y_pr),
                    recall_score(y_ts,y_pr), f1_score(y_ts,y_pr)])
        fpr,tpr,_=roc_curve(y_ts,y_proba)
        roc_data.append((n,fpr,tpr,auc(fpr,tpr)))
    st.dataframe(pd.DataFrame(res,columns=["Model","Acc","Prec","Rec","F1"]).set_index("Model").style.format("{:.2f}"))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### ROC Curves")
    import plotly.graph_objects as go
    fig=go.Figure()
    for n,fpr,tpr,aucv in roc_data:
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',name=f"{n} (AUC {aucv:.2f})"))
    fig.add_shape(type='line',x0=0,y0=0,x1=1,y1=1,line=dict(dash='dash'))
    fig.update_layout(xaxis_title="FPR",yaxis_title="TPR",template='plotly_white')
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --------- TAB 3: Clustering ---------
elif selected=="Clustering":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## K‑Means Clustering")
    X_clu = df[["EVs_%","Trucks_%"]]
    k = st.slider("Clusters",2,10,4)
    kmeans=KMeans(n_clusters=k,random_state=42).fit(X_clu)
    df["Cluster"]=kmeans.labels_
    st.plotly_chart(px.scatter(df,x="EVs_%",y="Trucks_%",color="Cluster",hover_data=["Company_Size","Maint_Strategy"], template="plotly_white"),use_container_width=True)
    st.write("Centers:", kmeans.cluster_centers_)
    st.markdown("</div>", unsafe_allow_html=True)

# --------- TAB 4: Association ---------
elif selected=="Association":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Association Rules: Key Benefits")
    trans = df["Key_Benefits"].str.split(", ")
    te=TransactionEncoder()
    onehot=te.fit(trans).transform(trans)
    df_one=pd.DataFrame(onehot,columns=te.columns_)
    sup=st.slider("Min Support",0.01,1.0,0.1,0.01)
    conf=st.slider("Min Confidence",0.1,1.0,0.6,0.05)
    freq=apriori(df_one,min_support=sup,use_colnames=True)
    rules=association_rules(freq,metric="confidence",min_threshold=conf).sort_values("confidence",ascending=False).head(10)
    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])
    st.markdown("</div>", unsafe_allow_html=True)
