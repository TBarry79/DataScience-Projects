"""
SCRIPT : generer_donnees_brutes.py
OBJECTIF : Créer un fichier Excel de 500 lignes (ventes fictives)
PROJET : P10 - Dashboard ventes PME
AUTEUR : Tidiane Barry
DATE : 2026-06-01
"""

# =====================================================
# 1. IMPORT DES BIBLIOTHEQUES
# =====================================================

import pandas as pd          
import numpy as np           
from datetime import datetime, timedelta  
import random                

# =====================================================
# 2. CONFIGURATION INITIALE
# =====================================================

# La graine aléatoire garantit que les résultats sont reproductibles
# (le même code donne les mêmes nombres aléatoires)
np.random.seed(42)

# Nombre de lignes à générer
nb_lignes = 500

# =====================================================
# 3. GÉNÉRATION DES DATES (du 01/01/2023 au 31/12/2024)
# =====================================================

# Date de début
date_debut = datetime(2023, 1, 1)

# Date de fin
date_fin = datetime(2024, 12, 31)

# Nombre total de jours entre la date de début et de fin
deltat = (date_fin - date_debut).days

# Création d'une liste de dates aléatoires
# Pour chaque vente, on choisit un jour aléatoire dans la période
dates = [date_debut + timedelta(days=random.randint(0, deltat)) for _ in range(nb_lignes)]

# Tri des dates par ordre chronologique
dates.sort()

# =====================================================
# 4. LISTE DES PRODUITS (nom, prix unitaire)
# =====================================================

produits = [
    ("Ordinateur portable", 850),      # (nom, prix)
    ("Écran 24 pouces", 180),
    ("Écran 27 pouces", 250),
    ("Serveur", 3200),
    ("Licence logiciel", 150),
    ("Clavier", 45),
    ("Souris", 25),
    ("Casque audio", 65),
    ("Tableau blanc", 120),
    ("Café", 8),
    ("Papeterie", 2),
    ("Formation Excel", 1200),
    ("Formation Power BI", 1000),
    ("Formation Python", 1500),
    ("Tableau de bord sur mesure", 2500)
]

# =====================================================
# 5. LISTE DES CLIENTS (id, nom, ville, segment)
# =====================================================

clients = [
    (101, "Durant", "Lyon", "Particulier"),
    (102, "Martinez", "Paris", "Pro"),
    (103, "Petit", "Marseille", "Particulier"),
    (104, "Bernard", "Lyon", "Collectivité"),
    (105, "Dubois", "Paris", "Particulier"),
    (106, "Thomas", "Lille", "Pro"),
    (107, "Robert", "Marseille", "Particulier"),
    (108, "Richard", "Paris", "Pro"),
    (109, "Moreau", "Lyon", "Collectivité"),
    (110, "Simon", "Lille", "Particulier"),
    (111, "Laurent", "Paris", "Pro"),
    (112, "Michel", "Marseille", "Collectivité"),
    (113, "Lefebvre", "Lyon", "Particulier"),
    (114, "Garcia", "Paris", "Pro"),
    (115, "David", "Lille", "Collectivité")
]

# =====================================================
# 6. GÉNÉRATION DES 500 VENTES
# =====================================================

# Liste vide qui va contenir les dictionnaires de données
data = []

# Boucle : on répète l'opération 'nb_lignes' fois (500)
for i in range(nb_lignes):
    
    # ---- Choix aléatoires ----
    
    # Choisir un client au hasard dans la liste
    client = random.choice(clients)
    
    # Choisir un produit au hasard dans la liste
    produit, prix_unitaire = random.choice(produits)
    
    # Quantité aléatoire entre 1 et 10
    quantite = random.randint(1, 10)
    
    # Calcul du montant total
    montant_total = quantite * prix_unitaire
    
    # ---- 2% de valeurs aberrantes (CA négatif) ----
    # Cela servira à tester le nettoyage des données plus tard
    if random.random() < 0.02:  # 0.02 = 2% des cas
        montant_total = -abs(montant_total)  # Rendre le montant négatif
    
    # ---- Statut de la commande ----
    # weights = probabilités : 85% Payé, 10% En attente, 5% Annulé
    statut = random.choices(
        ['Payé', 'En attente', 'Annulé'], 
        weights=[0.85, 0.10, 0.05]
    )[0]  # [0] car choices retourne une liste d'un élément
    
    # ---- Création du dictionnaire pour cette vente ----
    # Un dictionnaire = un objet avec des clés et des valeurs
    vente = {
        'vente_id': 20001 + i,              # ID unique (20001, 20002, ...)
        'date_vente': dates[i].strftime('%Y-%m-%d'),  # Date formatée
        'client_id': client[0],              # Premier élément du tuple
        'client_nom': client[1],             # Deuxième élément
        'client_ville': client[2],           # Troisième élément
        'client_segment': client[3],         # Quatrième élément
        'produit': produit,
        'quantite': quantite,
        'prix_unitaire_ht': prix_unitaire,
        'montant_total_ht': montant_total,
        'statut': statut
    }
    
    # Ajouter ce dictionnaire à la liste 'data'
    data.append(vente)

# =====================================================
# 7. CONVERSION EN DATAFRAME (tableau pandas)
# =====================================================

# pandas transforme la liste de dictionnaires en tableau structuré
df = pd.DataFrame(data)

# =====================================================
# 8. CRÉATION DU FICHIER EXCEL (avec plusieurs feuilles)
# =====================================================

# openpyxl est le moteur d'Excel pour pandas
with pd.ExcelWriter('Projet10_dashboard_ventes_pme/data/ventes_brutes_PME.xlsx', engine='openpyxl') as writer:
    
    # --- Feuille 1 : les données brutes ---
    df.to_excel(writer, sheet_name='ventes_brutes', index=False)
    # index=False signifie : ne pas ajouter la colonne des numéros de ligne
    
    # --- Feuille 2 : documentation (aide pour comprendre les colonnes) ---
    doc = pd.DataFrame({
        'Colonne': df.columns.tolist(),
        'Description': [
            'Identifiant unique de la vente (20001 à 20500)',
            'Date de la transaction (YYYY-MM-DD)',
            'ID du client (101 à 115)',
            'Nom du client',
            'Ville du client',
            'Segment client (Particulier/Pro/Collectivité)',
            'Produit vendu',
            'Quantité commandée',
            'Prix unitaire HT en euros',
            'Montant total HT = quantité × prix unitaire',
            'Statut de la commande (Payé/En attente/Annulé)'
        ]
    })
    doc.to_excel(writer, sheet_name='documentation', index=False)

# =====================================================
# 9. RAPPORT D'EXÉCUTION DANS LA CONSOLE
# =====================================================

print("=" * 60)
print("✅ FICHIER CREE AVEC SUCCÈS")
print("=" * 60)
print(f"📍 Emplacement : Projet10_dashboard_ventes_pme/data/ventes_brutes_PME.xlsx")
print(f"📊 Nombre de lignes : {len(df)}")
print(f"📋 Colonnes : {len(df.columns)}")

# Afficher la répartition des statuts
print("\n📊 Répartition des statuts :")
for statut, compteur in df['statut'].value_counts().items():
    print(f"   - {statut} : {compteur} ventes ({compteur/len(df)*100:.1f}%)")

# Afficher les valeurs aberrantes (CA négatif)
aberrantes = df[df['montant_total_ht'] < 0]
print(f"\n⚠️  Valeurs aberrantes (CA négatif) : {len(aberrantes)} ventes")
print("   (Ces lignes seront à nettoyer en semaine 2)")

# Aperçu des 5 premières lignes
print("\n📷 Aperçu des 3 premières lignes :")
print(df.head(3).to_string())

print("\n" + "=" * 60)
print("🎯 PROCHAIN ÉTAPE : Ajouter les feuilles 'clients' et 'produits' (Mardi)")
print("=" * 60)