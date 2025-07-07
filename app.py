from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model # type: ignore
import joblib
import numpy as np

app = Flask(__name__)

# Load models and preprocessing objects
crop_model = load_model('./models/crop_model.keras')
fertilizer_model = load_model('./models/fertilizer_model.keras')
rf_crop_model = joblib.load('./models/rf_crop_model.pkl')
rf_fertilizer_model = joblib.load('./models/rf_fertilizer_model.pkl')
scaler_crop = joblib.load('./models/scaler_crop.pkl')
scaler_fertilizer = joblib.load('./models/scaler_fertilizer.pkl')
encoder_crop = joblib.load('./models/encoder_crop.pkl')
fertilizer_label_encoder = joblib.load('./models/fertilizer_label_encoder.pkl')
soil_type_encoder = joblib.load('./models/soil_type_encoder.pkl')
crop_type_encoder = joblib.load('./models/crop_type_encoder.pkl')

# Crop mapping for fertilizer recommendation
crop_mapping = {
    'rice': 'rice',
    'maize': 'Maize',
    'chickpea': 'Pulses',
    'kidneybeans': 'kidneybeans',
    'pigeonpeas': 'Pulses',
    'mothbeans': 'Pulses',
    'mungbean': 'Pulses',
    'blackgram': 'Pulses',
    'lentil': 'Pulses',
    'pomegranate': 'pomegranate',
    'banana': 'pomegranate',
    'mango': 'pomegranate',
    'grapes': 'pomegranate',
    'watermelon': 'watermelon',
    'muskmelon': 'watermelon',
    'apple': 'pomegranate',
    'orange': 'orange',
    'papaya': 'pomegranate',
    'coconut': 'Oil seeds',
    'cotton': 'Cotton',
    'jute': 'Cotton',
    'coffee': 'coffee'
}

# NPK recommendations and fertilizer compositions
crop_original_recommendations = {
    'rice': {'N': '100-120 kg/ha', 'P': '40-50 kg/ha', 'K': '40-60 kg/ha'},
    'maize': {'N': '120-150 kg/ha', 'P': '50-60 kg/ha', 'K': '30-40 kg/ha'},
    'chickpea': {'N': '20-30 kg/ha', 'P': '40-50 kg/ha', 'K': '20-30 kg/ha'},
    'kidneybeans': {'N': '25-30 kg/ha', 'P': '50-60 kg/ha', 'K': '30-40 kg/ha'},
    'pigeonpeas': {'N': '20-25 kg/ha', 'P': '40-50 kg/ha', 'K': '20-30 kg/ha'},
    'mothbeans': {'N': '15-20 kg/ha', 'P': '30-40 kg/ha', 'K': '15-20 kg/ha'},
    'mungbean': {'N': '20-25 kg/ha', 'P': '40-50 kg/ha', 'K': '20-30 kg/ha'},
    'blackgram': {'N': '15-20 kg/ha', 'P': '30-40 kg/ha', 'K': '15-20 kg/ha'},
    'lentil': {'N': '25-30 kg/ha', 'P': '40-50 kg/ha', 'K': '20-30 kg/ha'},
    'pomegranate': {'N': '70-80 kg/ha', 'P': '30-40 kg/ha', 'K': '50-60 kg/ha'},
    'banana': {'N': '150-180 kg/ha', 'P': '50-60 kg/ha', 'K': '150-200 kg/ha'},
    'mango': {'N': '90-100 kg/ha', 'P': '30-40 kg/ha', 'K': '80-100 kg/ha'},
    'grapes': {'N': '100-120 kg/ha', 'P': '30-40 kg/ha', 'K': '140-160 kg/ha'},
    'watermelon': {'N': '70-80 kg/ha', 'P': '30-40 kg/ha', 'K': '70-80 kg/ha'},
    'muskmelon': {'N': '70-80 kg/ha', 'P': '30-40 kg/ha', 'K': '70-80 kg/ha'},
    'apple': {'N': '70-80 kg/ha', 'P': '30-40 kg/ha', 'K': '50-60 kg/ha'},
    'orange': {'N': '80-90 kg/ha', 'P': '30-40 kg/ha', 'K': '60-70 kg/ha'},
    'papaya': {'N': '90-100 kg/ha', 'P': '30-40 kg/ha', 'K': '90-100 kg/ha'},
    'coconut': {'N': '90-100 kg/ha', 'P': '40-50 kg/ha', 'K': '100-120 kg/ha'},
    'cotton': {'N': '70-80 kg/ha', 'P': '30-40 kg/ha', 'K': '30-40 kg/ha'},
    'jute': {'N': '40-50 kg/ha', 'P': '20-30 kg/ha', 'K': '20-30 kg/ha'},
    'coffee': {'N': '100-120 kg/ha', 'P': '30-40 kg/ha', 'K': '70-80 kg/ha'}
}

fertilizer_npk_composition = {
    'Urea': {'N': 46, 'P': 0, 'K': 0},
    'TSP': {'N': 0, 'P': 46, 'K': 0},  
    'Superphosphate': {'N': 0, 'P': 20, 'K': 0},
    'Potassium sulfate.': {'N': 0, 'P': 0, 'K': 50},
    'Potassium chloride': {'N': 0, 'P': 0, 'K': 60},
    'DAP': {'N': 18, 'P': 46, 'K': 0},  
    '28-28': {'N': 28, 'P': 28, 'K': 0},
    '20-20': {'N': 20, 'P': 20, 'K': 0},
    '17-17-17': {'N': 17, 'P': 17, 'K': 17},
    '15-15-15': {'N': 15, 'P': 15, 'K': 15},
    '14-35-14': {'N': 14, 'P': 35, 'K': 14},
    '14-14-14': {'N': 14, 'P': 14, 'K': 14},
    '10-26-26': {'N': 10, 'P': 26, 'K': 26},
    '10-10-10': {'N': 10, 'P': 10, 'K': 10}
}
# Efficiency factors for different soil types
soil_efficiency_factors = {
    'Sandy': {'N': 0.7, 'P': 0.5, 'K': 0.6},
    'Clayey': {'N': 1.2, 'P': 1.1, 'K': 1.3},
    'Red': {'N': 0.8, 'P': 0.7, 'K': 0.8},
    'Black': {'N': 1.0, 'P': 1.0, 'K': 1.1},
    'Loamy': {'N': 1.0, 'P': 1.0, 'K': 1.0}
}

def calculate_fertilizer(crop, soil_levels, fertilizer, soil_type):
    if crop not in crop_original_recommendations:
        raise ValueError(f"No NPK recommendations available for crop: {crop}")
    original_npk = crop_original_recommendations[crop]

    def parse_recommendation(value):
        value = value.replace("kg/ha", "").strip()
        if '-' in value:
            low, high = map(int, value.split('-'))
            return (low + high) / 2
        return int(value)

    recommendation_N = parse_recommendation(original_npk['N'])
    recommendation_P = parse_recommendation(original_npk['P'])
    recommendation_K = parse_recommendation(original_npk['K'])

    if fertilizer not in fertilizer_npk_composition:
        raise ValueError(f"No NPK composition available for fertilizer: {fertilizer}")
    fertilizer_npk = fertilizer_npk_composition[fertilizer]

    if soil_type not in soil_efficiency_factors:
        raise ValueError(f"No efficiency factors available for soil type: {soil_type}")
    efficiency = soil_efficiency_factors[soil_type]

    # Adjust recommendations based on soil efficiency
    adjusted_N = recommendation_N / efficiency['N']
    adjusted_P = recommendation_P / efficiency['P']
    adjusted_K = recommendation_K / efficiency['K']

    deficit_N = max(adjusted_N - soil_levels['N'], 0)
    deficit_P = max(adjusted_P - soil_levels['P'], 0)
    deficit_K = max(adjusted_K - soil_levels['K'], 0)

    fert_amount_N = deficit_N / (fertilizer_npk['N'] / 100) if fertilizer_npk['N'] > 0 else 0
    fert_amount_P = deficit_P / (fertilizer_npk['P'] / 100) if fertilizer_npk['P'] > 0 else 0
    fert_amount_K = deficit_K / (fertilizer_npk['K'] / 100) if fertilizer_npk['K'] > 0 else 0

    return max(fert_amount_N, fert_amount_P, fert_amount_K)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    crop_input = np.array([[  # Prepare input for crop prediction
        float(data['N']),
        float(data['P']),
        float(data['K']),
        float(data['Temperature']),
        float(data['Humidity']),
        float(data['pH']),
        float(data['Rainfall'])
    ]])

    scaled_crop_input = scaler_crop.transform(crop_input)

    rf_crop_probs = rf_crop_model.predict_proba(scaled_crop_input)
    cascade_crop_input = np.hstack((scaled_crop_input, rf_crop_probs))
    crop_prediction = crop_model.predict(cascade_crop_input)
    predicted_crop_index = np.argmax(crop_prediction, axis=1)[0]
    predicted_crop = encoder_crop.inverse_transform([predicted_crop_index])[0]
    mapped_crop = crop_mapping.get(predicted_crop, predicted_crop)

    fertilizer_input = np.array([[  
        float(data['N']),
        float(data['P']),
        float(data['K']),
        float(data['Temperature']),
        float(data['Humidity']),
        float(data['Moisture']),
        soil_type_encoder.transform([data['Soil Type']])[0],
        crop_type_encoder.transform([mapped_crop])[0]
    ]])

    fertilizer_input[:, :6] = scaler_fertilizer.transform(fertilizer_input[:, :6])

    rf_fertilizer_probs = rf_fertilizer_model.predict_proba(fertilizer_input)
    cascade_fertilizer_input = np.hstack((fertilizer_input, rf_fertilizer_probs))
    fertilizer_prediction = fertilizer_model.predict(cascade_fertilizer_input)
    predicted_fertilizer_index = np.argmax(fertilizer_prediction, axis=1)[0]
    predicted_fertilizer = fertilizer_label_encoder.inverse_transform([predicted_fertilizer_index])[0]
    soil_type = data['Soil Type']
    soil_levels = {'N': float(data['N']), 'P': float(data['P']), 'K': float(data['K'])}
    fertilizer_amount = round(calculate_fertilizer(predicted_crop, soil_levels, predicted_fertilizer,soil_type),2)

    return render_template(
        'index.html',
        predicted_crop=predicted_crop,
        predicted_fertilizer=predicted_fertilizer,
        fertilizer_amount=fertilizer_amount
    )

if __name__ == '__main__':
    app.run(debug=True)
