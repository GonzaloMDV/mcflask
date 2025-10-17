from flask import Flask, request, Response
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from flask_cors import CORS

app = Flask(__name__)

# --- Datos ---
recommendations_dict = {
    "otro": [
        "No se reconoce la lesión.",
        "Prueba de nuevo para identificar la lesión.",
        "No se puede identificar la lesión con certeza."
    ],
    "normal": [
        "No hay lesión visible en la piel.",
        "No se requieren primeros auxilios ni atención médica.",
        "La zona parece estar en condiciones normales; solo vigila cualquier cambio."
    ],
    "corte": [
        "Lava la herida con agua corriente y jabón antibacteriano durante al menos 5 minutos.",
        "Desinfecta con solución antiséptica como povidona yodada o alcohol isopropílico.",
        "Aplica una pomada antibiótica y cubre con un apósito estéril cambiándolo diariamente.",
        "Mantén la herida elevada para reducir la inflamación y evita actividades que puedan irritarla."
    ],
    "raspon": [
        "Limpia suavemente con agua tibia y jabón, evitando frotar la piel dañada.",
        "Aplica una crema hidratante calmante como aloe vera o vaselina para acelerar la cicatrización.",
        "Evita exponer la zona al sol directo hasta que se haya curado completamente.",
        "Si hay sangrado, aplica presión directa con un paño limpio durante 10-15 minutos."
    ],
    "moreton": [
        "Aplica compresas de hielo envueltas en un paño durante 15-20 minutos cada hora.",
        "Eleva la zona afectada por encima del nivel del corazón para reducir la hinchazón.",
        "Usa analgésicos de venta libre como ibuprofeno para aliviar el dolor y la inflamación.",
        "Masajea suavemente la zona después de 48 horas con aceite de ricino o crema anti-moretones."
    ],
    "quemadura": [
        "Enfría inmediatamente la quemadura bajo agua fría corriente durante 10-15 minutos.",
        "Cubre la zona con gasa esterilizada sin aplicar ninguna crema ni ungüento.",
        "Para quemaduras leves, aplica gel de aloe vera varias veces al día para aliviar el dolor.",
        "Busca atención médica urgente si la quemadura es profunda, mayor a 3 pulgadas de diámetro, o afecta manos, pies, cara o articulaciones."
    ],
    "picadura": [
        "Extrae el aguijón si está presente usando pinzas esterilizadas y lava la zona con agua y jabón.",
        "Aplica una pasta de bicarbonato de sodio y agua para neutralizar el veneno y reducir la comezón.",
        "Usa cremas antihistamínicas tópicas como benadryl para aliviar el picor e hinchazón.",
        "Monitorea signos de reacción alérgica severa como dificultad respiratoria, mareos o urticaria generalizada."
    ],
    "desmayo": [
        "Coloca al paciente en posición de Trendelenburg (piernas elevadas) y asegúrate de que respire adecuadamente.",
        "Brinda líquidos azucarados como jugo de naranja o bebidas deportivas para restaurar glucosa en sangre.",
        "Verifica signos vitales como pulso y respiración; si son irregulares, llama a emergencias.",
        "Permite que descanse en un lugar tranquilo y ventilado durante al menos 30 minutos antes de intentar levantarlo."
    ],
    "atragantamiento": [
        "Fomenta al paciente a toser vigorosamente mientras lo apoyas desde atrás.",
        "Realiza la maniobra de Heimlich: colócate detrás, abraza la cintura, y aplica cinco golpes rápidos entre los omóplatos seguidos de cinco compresiones abdominales.",
        "Si el objeto obstructor es visible en la boca, intenta retirarlo con cuidado usando pinzas esterilizadas.",
        "Llama al servicio de emergencias inmediatamente si el paciente pierde el conocimiento o no puede coughing."
    ]
}

category_features = {
    "otro": [0, 0, 0],
    "normal": [1, 1, 1],
    "corte": [3, 3, 2],
    "raspon": [2, 2, 1],
    "moreton": [2, 2, 1],
    "quemadura": [4, 4, 3],
    "picadura": [3, 3, 2],
    "desmayo": [5, 5, 1],
    "atragantamiento": [5, 5, 5]
}

# --- Modelo KNN ---
feature_matrix = np.array(list(category_features.values()))
categories = list(category_features.keys())
nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn_model.fit(feature_matrix)

# --- Función para obtener recomendación ---
def get_recommendation(user_vector):
    user_vector = np.array(user_vector).reshape(1, -1)
    distances, indices = nn_model.kneighbors(user_vector)
    nearest_category = categories[indices.flatten()[0]]
    recommendations = recommendations_dict[nearest_category]
    selected_rec = np.random.choice(recommendations)
    return {
        "category": nearest_category,
        "recommendation": selected_rec,
        "distance_score": float(distances.flatten()[0]),
        "all_options": recommendations
    }

# --- Endpoint Flask ---
@app.route("/recommend", methods=["POST"])
@app.route("/recommend", methods=["POST"])
def recommend_endpoint():
    data = request.json
    word = data.get("word", "").lower()
    
    if word not in category_features:
        return Response(
            json.dumps({"error": "Categoría no reconocida"}, ensure_ascii=False),
            mimetype='application/json'
        ), 400
    
    vector = category_features[word]
    result = get_recommendation(vector)
    
    return Response(
        json.dumps(result, ensure_ascii=False),
        mimetype='application/json'
    )


# --- Ejecutar servidor ---
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
CORS(app) 
# CORS(app, resources={r"/recommend": {"origins": "http://localhost:38517"}})
