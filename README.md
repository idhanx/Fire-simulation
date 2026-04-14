# 🔥 Forest Fire Prediction & Simulation using AI

An end-to-end AI system that predicts forest fire risk and simulates fire spread using geospatial data and machine learning.

---

## 📌 Overview

Uncontrolled forest fires pose a major threat to ecosystems and human life. This project builds a complete pipeline that:
* **Predicts fire-prone areas** using a CNN model
* **Generates a fire probability map** for the next day
* **Simulates fire spread over time** using Cellular Automata
* **Visualizes results** through an interactive Streamlit UI

---

## 🚀 Features

* **🔥 Fire Prediction Map**: Pixel-wise probability of fire occurrence.
* **🧠 CNN-based Model**: Lightweight and optimized for low-resource systems.
* **🌍 Geospatial Data Integration**: Land Use Land Cover (LULC) + weather data used as features.
* **🔁 Fire Spread Simulation**: Cellular Automata model with time progression.
* **🎬 Animation Support**: Fire spread visualized over time (1, 2, 3, 6, 12 hours).
* **📊 Confidence Map**: Highlights uncertainty in predictions.
* **🖥️ Streamlit UI**: Clean interface for demo and visualization.

---

## 🧱 Project Architecture

```text
Raw Data → Preprocessing → Feature Stack → CNN Model → Fire Prediction Map → CA Simulation → UI
```

---

## 📂 Project Structure

```text
forest_fire_ai/
│
├── data/
├── models/
├── training/
├── simulation/
├── inference/
├── ui/
├── utils/
├── outputs/
├── requirements.txt
└── main.py
```

---

## 🧠 Model Details

* **Model**: Convolutional Neural Network (CNN)
* **Input**: (2 × 512 × 512)
    * `Channel 1` → LULC
    * `Channel 2` → Weather
* **Output**: Fire probability map

### Optimizations Applied
* Spatial train/validation split (prevents leakage)
* Dropout (0.3) for regularization
* Label smoothing
* Input noise injection
* Class imbalance handling (weighted loss)
* Threshold tuning (precision-recall balanced)

---

## 🔥 Fire Spread Simulation

* Implemented using **Cellular Automata**
* **States**:
    * `0` → Unburned
    * `1` → Burning
    * `2` → Burned
* **Factors**:
    * Neighbor influence
    * Fire persistence
    * Probability-based spread

---

## 📊 Results

| Metric | Value |
| --- | --- |
| Accuracy | ~0.90 |
| Precision | ~0.90 |
| Recall | ~0.95 |
| F1 Score | ~0.92 |

> ⚠️ Note: High performance is influenced by limited and spatially correlated data. Future work includes multi-region datasets for better generalization.

---

## 🖥️ Run the Project

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train model**
   ```bash
   python training/train.py
   ```

3. **Evaluate model**
   ```bash
   python training/evaluate.py
   ```

4. **Run inference**
   ```bash
   python inference/predict.py
   ```

5. **Launch UI**
   ```bash
   streamlit run ui/app.py
   ```

---

## 🎥 Outputs
* Fire prediction heatmap
* Binary fire mask
* Confidence map
* Fire spread simulation (time-based)
* Animation (GIF)

---

## ⚠️ Limitations
* Uses single-region dataset
* No temporal sequence modeling (LSTM not integrated)
* Simulation is simplified (no wind/slope physics)

---

## 🚀 Future Improvements
* Integrate LSTM for temporal weather data
* Use real satellite datasets (VIIRS, MODIS)
* Add wind direction and terrain slope
* Export GeoTIFF for GIS tools
* Improve simulation realism

---

## 👨‍💻 Author

**Dhanush**  
B.Tech AI & Data Science  
Aspiring AI Engineer

---

## ⭐ Acknowledgment

This project demonstrates the application of AI in disaster prediction and highlights the importance of combining machine learning with simulation techniques.
