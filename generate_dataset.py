"""
Génère un dataset synthétique : titre + description de tâche -> effort estimé (heures)
Basé sur des grilles d'estimation logicielle réalistes.
"""
import pandas as pd
import random

random.seed(42)

# Chaque entrée : (titre, description, effort_heures_base)
tasks = [
    # --- CRUD simple (faible effort : 2h à 10h) ---
    ("Créer formulaire de contact", "Ajouter un formulaire simple avec nom, email et message, envoyé par mail", 6),
    ("Afficher liste des utilisateurs", "Page qui affiche la liste des utilisateurs avec pagination simple", 5),
    ("Ajouter bouton de suppression", "Bouton pour supprimer un élément avec une confirmation basique", 3),
    ("Modifier le profil utilisateur", "Formulaire pour éditer nom, email et photo de profil", 7),
    ("Page à propos statique", "Page HTML statique avec texte de présentation de l'entreprise", 2),
    ("Ajouter un champ de recherche", "Barre de recherche simple qui filtre une liste côté client", 4),
    ("Créer table des catégories", "CRUD basique pour gérer des catégories (ajout, édition, suppression)", 8),
    ("Affichage des commentaires", "Liste des commentaires sous un article, sans modération", 5),
    ("Ajout d'un menu de navigation", "Menu responsive avec liens vers les pages principales", 4),
    ("Changer le mot de passe", "Formulaire simple de changement de mot de passe avec validation basique", 5),
    ("Page de contact avec carte", "Formulaire de contact avec une carte Google Maps intégrée", 6),
    ("Liste des produits avec filtres simples", "Affichage produits avec filtre par catégorie uniquement", 7),
    ("Ajout d'un système de favoris", "Bouton pour ajouter/retirer un élément des favoris de l'utilisateur", 6),
    ("Export CSV simple", "Bouton pour exporter une liste en fichier CSV basique", 5),
    ("Page 404 personnalisée", "Page d'erreur personnalisée avec redirection vers l'accueil", 2),
    
    # --- Complexité modérée (10h à 25h) ---
    ("Authentification utilisateur", "Système de login et inscription avec validation email et hashage du mot de passe", 16),
    ("Système de notifications", "Notifications en base de données affichées dans une cloche avec compteur non lus", 18),
    ("Upload et gestion d'images", "Permettre l'upload d'images avec redimensionnement et stockage sécurisé", 14),
    ("Dashboard avec statistiques", "Tableau de bord affichant des graphiques et indicateurs clés à partir de la base de données", 20),
    ("Système de rôles et permissions", "Gestion de rôles (admin, manager, user) avec permissions différenciées par page", 22),
    ("Recherche avancée avec filtres multiples", "Recherche avec plusieurs filtres combinés (date, catégorie, statut, prix)", 15),
    ("Système de commentaires avec modération", "Commentaires avec système de signalement et validation par un admin", 17),
    ("Export PDF de rapport", "Génération de rapports PDF dynamiques avec mise en page personnalisée", 13),
    ("Système de tags et étiquettes", "Ajout d'un système de tags réutilisables sur plusieurs entités", 12),
    ("Historique des actions utilisateur", "Journal (log) qui trace les actions importantes effectuées par chaque utilisateur", 14),
    ("Système de messagerie interne", "Chat simple entre utilisateurs avec historique des messages, sans temps réel", 20),
    ("Calendrier des événements", "Calendrier interactif affichant les événements avec vue mois/semaine", 18),
    ("Gestion des fichiers joints", "Permettre l'ajout de plusieurs fichiers (pdf, image) à une tâche avec prévisualisation", 15),
    ("Système de validation en plusieurs étapes", "Workflow d'approbation à plusieurs niveaux (soumission, revue, validation finale)", 19),
    ("Intégration d'un système de tags dynamiques", "Auto-complétion de tags existants lors de la saisie", 11),
    
    # --- Complexité élevée (30h à 50h) ---
    ("Intégration paiement en ligne", "Ajouter Stripe pour permettre aux clients de payer leurs commandes, avec gestion des erreurs et webhook de confirmation", 40),
    ("Système de messagerie en temps réel", "Chat en temps réel entre utilisateurs avec WebSockets, indicateurs de frappe et statut en ligne", 45),
    ("Module de reporting avancé avec export multi-format", "Génération de rapports complexes avec agrégations, filtres dynamiques et export PDF/Excel", 38),
    ("Système de recommandation de produits", "Moteur de recommandation basé sur l'historique d'achat et le comportement de l'utilisateur", 50),
    ("Intégration API de paiement multiple", "Support de plusieurs passerelles de paiement (Stripe, PayPal, virement) avec réconciliation", 42),
    ("Système d'authentification à deux facteurs", "Ajout de la 2FA par SMS ou application d'authentification avec gestion de la récupération", 30),
    ("Synchronisation temps réel multi-utilisateurs", "Édition collaborative en temps réel avec gestion des conflits entre plusieurs utilisateurs", 48),
    ("Système de recherche full-text avec Elasticsearch", "Intégration d'un moteur de recherche performant avec indexation et pertinence des résultats", 35),
    ("Module de facturation automatique récurrente", "Génération automatique de factures récurrentes avec gestion des abonnements et relances", 36),
    ("Système de géolocalisation en temps réel", "Suivi de position en temps réel sur une carte avec historique des trajets", 40),
    ("Intégration webhook multi-services", "Système d'événements sortants vers plusieurs services externes avec retry automatique", 32),
    ("Module d'analyse prédictive", "Tableau de bord utilisant un modèle prédictif pour anticiper des tendances métier", 44),
    ("Système de vidéoconférence intégrée", "Appels vidéo entre utilisateurs directement dans l'application avec partage d'écran", 55),
    ("Architecture microservices pour module critique", "Refonte d'un module en microservice indépendant avec communication asynchrone", 46),
    ("Système de cache distribué", "Mise en place de Redis pour la mise en cache distribuée sur plusieurs instances serveur", 28),
    
    # --- Très complexe / IA / temps réel (50h à 70h) ---
    ("Intelligence artificielle de détection de fraude", "Modèle de machine learning pour détecter les transactions frauduleuses en temps réel", 60),
    ("Chatbot intelligent avec NLP", "Assistant conversationnel utilisant le traitement du langage naturel pour répondre aux questions clients", 55),
    ("Système de reconnaissance d'image", "Intégration d'un modèle de vision par ordinateur pour classifier des images uploadées", 58),
    ("Estimation automatique de l'effort par IA", "Modèle de machine learning qui prédit l'effort nécessaire à partir de la description d'une tâche", 52),
    ("Système de recommandation basé sur le deep learning", "Moteur de recommandation utilisant un réseau de neurones entraîné sur les données utilisateurs", 62),
    ("Plateforme temps réel multi-tenant à grande échelle", "Architecture supportant plusieurs organisations simultanément avec isolation des données et scalabilité", 65),
    ("Système de traduction automatique intégré", "Traduction en temps réel du contenu de l'application dans plusieurs langues via un modèle NLP", 50),
    ("Moteur de scoring prédictif client", "Modèle prédictif qui score les clients selon leur probabilité de conversion", 54),
    ("Système de sécurité avec détection d'anomalies", "Surveillance en temps réel du système avec détection d'anomalies via machine learning", 56),
    ("Automatisation complète des tests avec IA", "Génération automatique de scénarios de test à partir de l'analyse du code par IA", 48),
]

def augment_variations(tasks, n_target=600):
    """Ajoute des variations légères (synonymes, reformulations) pour enrichir le dataset à 600 lignes."""
    prefixes = ["", "Développer : ", "Mettre en place ", "Ajouter la fonctionnalité : ", "Créer "]
    rows = []
    for title, desc, effort in tasks:
        rows.append((title, desc, effort))
    
    i = 0
    while len(rows) < n_target:
        base_title, base_desc, base_effort = tasks[i % len(tasks)]
        prefix = random.choice(prefixes)
        noise = random.uniform(0.85, 1.15)  # variation réaliste de +/-15%
        new_title = f"{prefix}{base_title}".strip()
        rows.append((new_title, base_desc, round(base_effort * noise, 1)))
        i += 1
    return rows

# GÉNÉRATION DU DATASET (600 lignes pour une meilleure précision du modèle)
all_rows = augment_variations(tasks, n_target=600)
random.shuffle(all_rows)

df = pd.DataFrame(all_rows, columns=["title", "description", "effort_hours"])
df["text"] = df["title"] + " " + df["description"]
df = df[["title", "description", "text", "effort_hours"]]

# Sauvegarde locale (chemin relatif - fonctionne sur Windows)
df.to_csv("tasks_dataset.csv", index=False, encoding="utf-8")

print(f"✅ Dataset généré avec succès : {len(df)} lignes")
print("\n👀 Aperçu des données (5 premières lignes) :")
print(df.head())
print("\n📊 Statistiques des efforts (heures) :")
print(df["effort_hours"].describe())