"""
Entraîne le modèle NLP : TF-IDF (texte) + Random Forest (régression) -> effort en heures
Compatible Windows et Linux (chemins relatifs)
"""
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Charger le dataset (avec vérification de sécurité)
dataset_path = "tasks_dataset.csv"
if not os.path.exists(dataset_path):
    print(f"❌ ERREUR : Le fichier '{dataset_path}' est introuvable.")
    print("💡 Solution : Lance d'abord la commande 'python generate_dataset.py'")
    exit()

print("🔄 Chargement du dataset en cours...")
df = pd.read_csv(dataset_path)
X = df["text"]
y = df["effort_hours"]

# 2. Split train/test (80% entraînement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Pipeline : TF-IDF -> Random Forest
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=300,
        ngram_range=(1, 2),      # Capture les mots seuls ET les expressions ("temps réel")
        stop_words=None,          # On garde tous les mots pour le français
        lowercase=True,
    )),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1                 # Utilise tous les cœurs du CPU pour aller plus vite
    )),
])

# 4. Entraînement
print("🧠 Entraînement du modèle en cours (cela peut prendre quelques secondes)...")
pipeline.fit(X_train, y_train)

# 5. Évaluation (Les résultats "comme sur Kaggle")
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")

print("\n" + "=" * 60)
print("📊 RÉSULTATS DU MODÈLE (Comme sur Kaggle)")
print("=" * 60)
print(f"MAE (erreur moyenne en heures) : {mae:.2f}h")
print(f"R² (test set)                  : {r2:.3f}")
print(f"R² (cross-validation 5-fold)   : {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
print("=" * 60)

# 6. Exemples de prédictions
print("\n🔍 Exemples de prédictions vs réel :")
comparison = pd.DataFrame({
    "Texte de la tâche": [str(x)[:60] + "..." if len(str(x)) > 60 else x for x in X_test.values[:5]],
    "Réel (heures)": y_test.values[:5],
    "Prédit (heures)": [round(p, 1) for p in y_pred[:5]],
})
print(comparison.to_string(index=False))

# 7. Sauvegarder le modèle (chemin relatif, compatible Windows)
model_path = "nlp_effort_model.pkl"
joblib.dump(pipeline, model_path)
print(f"\n✅ Modèle sauvegardé avec succès sous : {model_path}")