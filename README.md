
# 🌾 AgroBalance – Smart Nutrient Management System

AgroBalance is a smart agricultural web application designed to assist farmers in determining the ideal balance of nutrients required for their crops based on soil conditions. It serves as a decision-support tool that helps optimize fertilizer usage, ensuring healthier crops, sustainable farming practices, and increased yields.

## 🚀 Features

- 📊 **Nutrient Prediction**: Accurately predicts required nitrogen (N), phosphorus (P), and potassium (K) levels based on soil data.
- 🧠 **Machine Learning Model**: Uses a trained ML model to analyze soil parameters and recommend NPK values.
- 💬 **User-Friendly Interface**: Clean, intuitive UI for easy interaction by farmers and agronomists.
- 🌱 **Crop-wise Insights**: Recommendations vary based on crop type and soil health, enabling customized treatment.

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Flask)
- **Machine Learning**: Scikit-learn
- **Model**: Trained on open-source soil data for nutrient prediction
- **Deployment**: Localhost / Flask-based server

## 🧪 How It Works

1. Users input soil features like pH, temperature, humidity, rainfall, etc.
2. The backend ML model processes this data to predict the optimal N, P, K values.
3. The application displays the recommended nutrient levels for balanced crop growth.

## 📂 Project Structure

```
AgroBalance/
│
├── static/              # CSS, JS, and image files
├── templates/           # HTML templates (Jinja2)
├── model/               # Saved ML model
├── app.py               # Flask app entry point
├── README.md            # Project documentation
└── requirements.txt     # Dependencies
```

## 📌 Installation & Usage

1. **Clone the repository**  
```bash
git clone https://github.com/BhavanaMinchu/AgroBalance.git
cd AgroBalance
```

2. **Create virtual environment & activate**  
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install dependencies**  
```bash
pip install -r requirements.txt
```

4. **Run the app**  
```bash
python app.py
```

5. **Visit on browser**  
```
http://127.0.0.1:5000/
```

## 📚 Dataset

This project uses publicly available soil and crop nutrient datasets (e.g., NPK prediction datasets) for model training. You can update the model using newer or region-specific data.

## 🤝 Contributions

Contributions are welcome! Feel free to fork this repository, make changes, and open a pull request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

> ⚠️ Disclaimer: This tool is a prototype and should be used for educational or assistive purposes. Always consult local agronomists before making large-scale soil treatments.
