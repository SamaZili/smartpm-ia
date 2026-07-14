import gradio as gr
import joblib

# Charger le modèle
model = joblib.load("nlp_effort_model.pkl")

def predict_effort(title, description):
    if not title.strip() and not description.strip():
        return "⚠️ Veuillez saisir au moins un titre ou une description."
    
    text = f"{title} {description}".strip()
    prediction = model.predict([text])[0]
    prediction = max(0, round(float(prediction), 2))
    days = round(prediction / 8, 1)
    
    if prediction < 15:
        complexity = "🟢 Simple"
    elif prediction < 35:
        complexity = "🟡 Modérée"
    elif prediction < 55:
        complexity = " Élevée"
    else:
        complexity = "🔴 Très complexe / IA"

    return f"⏱️ Effort estimé : {prediction} heures\n📅 ≈ {days} jours ouvrés\n🎯 Complexité : {complexity}"

# Créer l'interface
interface = gr.Interface(
    fn=predict_effort,
    inputs=[
        gr.Textbox(label="Titre de la tâche", placeholder="Ex: Intégration paiement en ligne"),
        gr.Textbox(label="Description", placeholder="Ex: Ajouter Stripe avec gestion des erreurs...", lines=3)
    ],
    outputs=gr.Textbox(label="Résultat de l'IA"),
    title="SmartPM - Test d'Estimation IA",
    description="Teste ton modèle NLP localement avant de l'intégrer à Laravel."
)

if __name__ == "__main__":
    interface.launch()