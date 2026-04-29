<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Trebuchet+MS&size=32&pause=1000&color=00BFC4&center=true&vCenter=true&width=900&lines=EV+Infrastructure+Optimization+%26+Planning;Decision-Support+Tools+for+Smart+Cities;ML+%2B+MCDM+%2B+Data+Warehousing+Pipeline" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![DuckDB](https://img.shields.io/badge/DuckDB-Data_Warehouse-FFF000?style=for-the-badge&logo=duckdb&logoColor=black)](https://duckdb.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-87%25_Accuracy-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Folium](https://img.shields.io/badge/Folium-Interactive_Maps-77B829?style=for-the-badge)](https://python-visualization.github.io/folium/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **A complete end-to-end data science pipeline** — from raw EV charging data to live, interactive deployment maps — covering all 6 DWDM modules, 10 pipeline phases, and full Indian city adaptation.

<br/>

**DWDM — CSE4005 · VIT-AP University · April 2026**

*Faculty: Prof. Gosala Bethany · Sabudh Foundation*

| 👤 Arpit Makkar · `23BCE7565` | 👤 Tripjot Singh · `23BCE8227` |
|:---:|:---:|

<br/>

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results-at-a-glance)
- [System Architecture](#-system-architecture--10-phase-pipeline)
- [Dataset](#-dataset--hero)
- [Data Warehouse](#-module-1--data-warehouse)
- [Machine Learning](#-module-4--classification--prediction)
- [SHAP Explainability](#-shap-explainability)
- [Clustering](#-module-5--cluster-analysis)
- [Pattern Mining](#-module-3--frequent-pattern-mining)
- [Outlier Detection](#-module-6--outlier-detection)
- [MCDM Site Selection](#-mcdm-multi-criteria-decision-making)
- [Indian City Adaptation](#-indian-city-adaptation)
- [Live Deployments](#-live-deployments)
- [Repository Structure](#-repository-structure)
- [Setup & Installation](#-setup--installation)
- [DWDM Syllabus Coverage](#-100-dwdm-syllabus-coverage)

---

## 🌐 Overview

This project builds a **city-agnostic, decision-support framework** for optimal EV charging station placement. It fuses **machine learning demand forecasting**, **multi-criteria decision making (MCDM)**, and **interactive geospatial visualisation** into a single, reproducible pipeline — validated on Hong Kong (HERO dataset) and adapted for **Hyderabad & Bengaluru, India**.

```
Raw Geospatial Data  →  DuckDB Warehouse  →  ML Ensemble (87%)  →  MCDM Ranking  →  Live Maps
      613K rows              Star Schema         SHAP Explained        3 Scenarios     3 Deployments
```

The framework is **genuinely transferable**: the same XGBoost + Random Forest ensemble achieves ~83% on Indian city data with **zero architecture changes** — only MCDM weights require recalibration for India's two-wheeler dominated mobility landscape.

---

## 🏆 Key Results At a Glance

<div align="center">

| Metric | Value |
|:---|:---|
| 📦 Dataset rows processed | **613,109 rows · 44 features** |
| 🤖 Ensemble model accuracy | **~87% (Hong Kong) · ~83% (India)** |
| 📍 Sites recommended per scenario | **300 sites × 3 scenarios** |
| 📚 DWDM syllabus coverage | **100% — all 6 modules** |
| 🗺️ Live deployments | **Folium · Deck.gl · Streamlit** |
| 🇮🇳 Indian cities adapted | **Hyderabad + Bengaluru** |
| 🚗 India MCDM scenarios | **Two-Wheeler · FAME-II · IT Corridor** |
| 🔑 Top demand driver (SHAP) | **Residential POI density \|φ\| = 0.048** |

</div>

---

## 🏗 System Architecture — 10-Phase Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE 10-PHASE PIPELINE                           │
├──────────┬──────────────────────────┬───────────────────────────────────┤
│  Module  │  Phase                   │  Key Output                       │
├──────────┼──────────────────────────┼───────────────────────────────────┤
│ Module 1 │ 1. Data Warehouse        │ Star schema · DuckDB · OLAP ops   │
│ Module 1 │ 2. Preprocessing         │ 380K zones removed · 3 features   │
│ Module 2 │ 3. EDA + Correlation     │ Pearson/Spearman · r = +0.41      │
│ Module 6 │ 4. Outlier Detection      │ IForest + LOF · 78% agreement    │
│ Module 3 │ 5. Apriori Rules         │ Lift = 1.50 ★ top association     │
│ Module 2 │ 6. Feature Selection     │ RF importance · top 8 retained    │
│ Module 4 │ 7. 4-Model Benchmark     │ XGB+RF Ensemble 87%               │
│ Module 5 │ 8. Cluster Analysis      │ DBSCAN dedup · K-Means 4 types    │
│ MCDM     │ 9. 3 Scenarios           │ Profit / Equity / Balanced        │
│ Outputs  │ 10. 3 Deployments        │ Folium · Deck.gl · Streamlit      │
└──────────┴──────────────────────────┴───────────────────────────────────┘
```

---

## 📊 Dataset — HERO

The project uses the **HERO (Hong Kong EV Road Occupancy)** dataset — a large, real-world EV charging demand dataset.

| Property | Detail |
|:---|:---|
| **Rows** | 613,109 |
| **Columns** | 44 features |
| **Domain** | EV charging demand · Hong Kong road network |
| **Key features** | Residential POI density, Commercial POI density, Road centrality, Transit hub proximity, Shannon entropy |
| **Preprocessing** | 380K ocean/empty zones removed · MinMaxScaler normalisation · 3 features engineered |
| **Zero-inflation** | 21 sparse features identified and handled via EDA |

---

## 🗄 Module 1 — Data Warehouse

A **full DuckDB star schema** was designed from scratch — not just loaded into a flat table.

```
fact_demand
    ├── dim_location   (grid_id, lat, lng, zone_type, city)
    ├── dim_poi        (residential_density, commercial_density, transit_proximity)
    └── dim_time       (hour, day_type, season)
```

**OLAP operations implemented:**
- `ROLLUP` — aggregate demand from grid → district → city
- `DRILL-DOWN` — decompose city totals to zone-type level
- `PIVOT` — cross-tab demand by zone type × hour of day
- Indian grid registered with zone-type dimension: *IT Corridor, Old City, Residential, Transit Hub*

---

## 🤖 Module 4 — Classification & Prediction

Four classifiers were benchmarked on a binary high/low demand prediction task:

| Model | Accuracy | F1 Score | Notes |
|:---|:---:|:---:|:---|
| Decision Tree | ~72% | ~0.71 | Baseline — interpretable, max_depth=8 |
| Naïve Bayes | ~63% | ~0.61 | Weakest — assumes feature independence |
| Random Forest | ~84% | ~0.83 | Strong solo, 100 trees |
| **XGB + RF Ensemble** ★ | **~87%** | **~0.87** | **Winner — soft-voting, all metrics best** |

The **XGB + Random Forest soft-voting ensemble** outperforms a 33% random baseline by over 54 percentage points and beats each individual model. The same ensemble retrained on Indian data achieves **~83%** with zero architecture changes.

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) was integrated to make the black-box ensemble interpretable for urban planners.

**Top findings:**

```
Residential POI density   |φ| = 0.048  ← #1 driver  (counterintuitive: EV users charge near home)
Commercial POI density    |φ| = 0.022
Road centrality           |φ| = 0.018
Transit hub proximity     |φ| = 0.015
```

> **Key insight:** Residential density outweighs commercial density by 2× — EV users primarily charge at or near their residences, not at workplaces. This has direct policy implications for where stations should be sited.

---

## 🗺️ Module 5 — Cluster Analysis

Two clustering algorithms were combined to identify **urban charging archetypes**:

**DBSCAN** (eps = 0.005, ~500m radius for HK · 0.006 for India)
- Used for **geographic deduplication** — merged candidate sites within 500m of each other
- Removed noise points (isolated candidate sites with no nearby demand)

**K-Means** (k = 4, elbow + silhouette validated)
- Identified **4 Hong Kong archetypes**: Urban Core · Residential Zone · Transit Hub · Suburban Edge
- Identified **5 Indian archetypes**: IT Corridor · Old City / Bazaar · Residential · Transit Hub · Suburban

---

## 🔗 Module 3 — Frequent Pattern Mining

Apriori algorithm applied to **POI co-occurrence** patterns across demand grid cells.

```python
# Configuration
min_support    = 0.15
min_confidence = 0.60
min_lift       = 1.0
```

**Top rule discovered:**

```
{Commercial POI, Residential POI}  →  {High EV Demand}
    support = 0.19 · confidence = 0.74 · lift = 1.50  ★
```

On Indian data, `{Transit, Residential}` co-occurrence was found in **top demand areas** — reinforcing the two-wheeler charging pattern near residential-transit corridors.

---

## 🚨 Module 6 — Outlier Detection

Two complementary outlier detection algorithms were applied to identify anomalous grid cells:

| Method | Configuration | Role |
|:---|:---|:---|
| **Isolation Forest** | contamination = 5% | Global anomaly detection |
| **LOF** (Local Outlier Factor) | k = 20 neighbours | Local density anomalies |

**78% agreement** between methods — cells flagged by both were definitively removed. PCA projection confirmed spatial clustering of outliers in ocean/uninhabited zones. Applied to Indian data to remove anomalous uninhabited cells before modelling.

---

## ⚖️ MCDM — Multi-Criteria Decision Making

Three planning scenarios were designed, each producing **300 ranked EV station sites**:

<div align="center">

| Scenario | Residential | Commercial | Transit | Road | Centrality |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Profit** | 5% | 40% | 5% | 30% | 20% |
| **Equity** | 35% | 10% | 25% | 20% | 10% |
| **Balanced** | 20% | 25% | 15% | 25% | 15% |

</div>

**Key finding:** Profit and Equity scenarios select dramatically different site profiles — **78% Urban Core** (Profit) vs **64% Residential Zone** (Equity) — confirming the weight system meaningfully captures different planning priorities.

---

## 🇮🇳 Indian City Adaptation

The framework was adapted for **Hyderabad and Bengaluru** with six structural changes:

**1. Two-wheeler dominance** — India's 80:20 two-to-four-wheeler ratio demands residential weight 5% → 35% and transit hub weight 5% → 25%.

**2. Old City / Bazaar zone type** — A new zone type was added with high entropy signature to capture bazaar-area charging demand (no HK equivalent).

**3. IT Corridor scenario** — Commercial + primary road + centrality signature unique to Hyderabad/Bengaluru tech parks; a dedicated MCDM scenario was designed.

**4. FAME-II policy layer** — India's government scheme mandates public and shared transport corridor priority; an entire new scenario was built around FAME-II eligibility criteria.

**5. Residential peak shift** — Indian charging peaks at 7–9 AM (morning commute) and evening home-return, unlike HK's 9–5 PM commercial peak.

**6. OSMnx + synthetic pipeline** — Both a real OSMnx road network pipeline and a synthetic grid fallback were implemented for environments without full internet access.

### India MCDM Scenario Weights

| Scenario | Residential | Transit Hub | Commercial | Road | Centrality |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Two-Wheeler Priority** | 35% | 25% | 10% | 20% | 10% |
| **FAME-II Equity** | 30% | 30% | 10% | 20% | 10% |
| **IT Corridor** | 15% | 15% | 30% | 25% | 15% |

---

## 🚀 Live Deployments

### 1. 🗺️ Folium Interactive Map (Hong Kong + India)

- AI demand heatmap filtered to top 75% demand zones
- Three India-specific scenario layers (toggle on/off)
- Zone-type popups: zone type, AI demand %, MCDM score, city context
- Same Folium architecture for HK and India — zero code change

### 2. 🌐 Deck.gl 3D Visualisation

- 3D extruded demand columns by grid cell
- Arc layer: residential zones → nearest recommended station
- Deployed as static HTML via GitHub Pages

### 3. 📊 Streamlit Dashboard

- Interactive scenario switcher (Profit / Equity / Balanced)
- SHAP waterfall chart for individual site explanations
- India variant with FAME-II eligibility filter
- Side-by-side HK vs India comparison

---

## 📁 Repository Structure

```text
📦 EV-Infrastructure-Optimization-and-Planning
 ┣ 📂 Dataset-Visualizations     # Contains charts, plots, and graphical EDA of EV data
 ┣ 📂 Visualization-Maps         # Geographical maps and interactive visualizations
 ┣ 📂 model                      # Exported machine learning/optimization models
 ┣ 📂 notebooks                  # Jupyter Notebooks for EDA, preprocessing, and model training
 ┣ 📜 index.html                 # Main HTML file for viewing web-based visualizations
 ┗ 📜 README.md                  # Project documentation (You are here!)
```

---

## ⚙️ Setup & Installation

### Prerequisites

```bash
Python 3.10+
```

### 1. Clone the repository

```bash
git clone https://github.com/ArpitMakkar12/EV-Infrastructure-Optimization-and-Planning.git
cd EV-Infrastructure-Optimization-and-Planning
```

### 2. Install dependencies

```bash
pip install duckdb xgboost scikit-learn shap folium streamlit mlxtend \
            osmnx geopandas pandas numpy matplotlib seaborn plotly
```

### 3. Run the pipeline (Jupyter)

```bash
jupyter notebook notebooks/
```

Run notebooks in order `01_` → `08_` for the full end-to-end pipeline.

### 4. Launch Streamlit dashboard

```bash
streamlit run app.py
```

### 5. View the Deck.gl 3D map

Open `index.html` directly in a browser or visit the [GitHub Pages deployment](https://arpitmakkar12.github.io/EV-Infrastructure-Optimization-and-Planning/).

---

## 📚 100% DWDM Syllabus Coverage

| Module | Topic | HK Implementation | India Implementation |
|:---|:---|:---|:---|
| **Module 1** | Data Warehousing | Star schema · DuckDB · ROLLUP / DRILL-DOWN / PIVOT | Indian grid · OLAP by zone type |
| **Module 1** | Data Preprocessing | 380K zones removed · Shannon entropy · MinMaxScaler | Zone filter · two-wheeler proxy feature |
| **Module 2** | Data Mining Concepts | Pearson + Spearman · zero-inflation report | Feature comparison HK vs India |
| **Module 3** | Frequent Pattern Mining | Apriori · lift = 1.50 · min_support = 0.15 | Transit + Residential co-occurrence |
| **Module 4** | Classification & Prediction | DT / NB / RF / XGB+RF · 87% · SHAP | Same ensemble · ~83% · transferability confirmed |
| **Module 5** | Cluster Analysis | DBSCAN dedup · K-Means 4 archetypes | 5 Indian archetypes · eps = 0.006 |
| **Module 6** | Outlier Analysis | Isolation Forest + LOF · 78% agreement · PCA | IForest on Indian grid · uninhabited removal |

---

## 🔭 Future Work

- **Real Indian OSM data** — Pull Hyderabad/Bengaluru road network via OSMnx; validate against BESCOM + TSSPDCL station locations
- **Temporal demand modelling** — Hour-of-day patterns, seasonal monsoon impact, time-series OLAP in DuckDB
- **Advanced visualisation** — Animated arc layers (residential zones → nearest station) on a deployed Deck.gl 3D Indian city map
- **REST API + scaling** — Wrap pipeline as an API where any city POSTs a bounding box and receives ranked sites back; scale to Mumbai, Delhi, Chennai, Pune, Ahmedabad

---

## 📜 Citation

If you use this framework or reference this work, please cite:

```bibtex
@project{ev_infrastructure_2026,
  title   = {Decision-Support Tools for EV Infrastructure Planning},
  author  = {Makkar, Arpit and Singh, Tripjot},
  course  = {DWDM CSE4005, VIT-AP University},
  advisor = {Gosala Bethany},
  year    = {2026},
  url     = {https://github.com/ArpitMakkar12/EV-Infrastructure-Optimization-and-Planning}
}
```

---

<div align="center">

**Built with 🔋 for smarter cities · VIT-AP University × Sabudh Foundation · April 2026**

[![GitHub stars](https://img.shields.io/github/stars/ArpitMakkar12/EV-Infrastructure-Optimization-and-Planning?style=social)](https://github.com/ArpitMakkar12/EV-Infrastructure-Optimization-and-Planning/stargazers)

</div>
