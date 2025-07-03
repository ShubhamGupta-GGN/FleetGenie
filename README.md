# FleetGenie Predictive Maintenance Dashboard

A Streamlit dashboard showcasing data visualisation and machine‑learning capabilities on a synthetic fleet‑maintenance survey dataset.

## Features

* **Data Visualisation** – 10 complex descriptive insights with interactive filters  
* **Classification** – KNN, Decision Tree, Random Forest, Gradient Boosting (metrics table, confusion matrix toggle, ROC curves, upload new data, download predictions)  
* **Clustering** – K‑Means with elbow chart, dynamic slider (2‑10 clusters), persona table, download labelled data  
* **Association Rules** – Apriori on selected columns with parameter controls, top‑10 rules by confidence  
* **Regression Insights** – Linear, Ridge, Lasso, Decision‑Tree regressors with quick insights

## Quick Start (local)

```bash
git clone <your‑repo‑url>
cd fleetgenie_dashboard
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy to Streamlit Cloud

1. Push this folder to a new GitHub repo.  
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and **New App** → point to your repo.  
3. Set **Main file** to `streamlit_app.py`.  
4. Add `requirements.txt` (already included).  
5. Deploy!

The app will automatically pull the dataset from the repo’s `data/` folder.
