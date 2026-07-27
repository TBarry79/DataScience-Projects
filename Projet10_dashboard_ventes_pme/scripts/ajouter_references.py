"""
SCRIPT : ajouter_references.py
OBJECTIF : Ajouter les feuilles 'clients' et 'produits' au fichier Excel
PROJET : P10 - Dashboard ventes PME
AUTEUR : Tidiane Barry
DATE : 2026-06-02
"""

# =====================================================
# 1. IMPORT DES BIBLIOTHEQUES
# =====================================================

import pandas as pd
from openpyxl import load_workbook

# =====================================================
# 2. DÉFINITION DES DONNÉES CLIENTS (15 clients)
# =====================================================

# Liste des clients avec leurs caractéristiques
# Format : (id, nom, ville, segment, date_premiere_commande, ca_total)
clients_data = [
    (101, "Durant", "Lyon", "Particulier", "2023-01-15", 2970.50),
    (102, "Martinez", "Paris", "Pro", "2023-02-20", 21670.00),
    (103, "Petit", "Marseille", "Particulier", "2023-03-10", 295.00),
    (104, "Bernard", "Lyon", "Collectivité", "2023-01-05", 12470.00),
    (105, "Dubois", "Paris", "Particulier", "2023-04-22", 5405.00),
    (106, "Thomas", "Lille", "Pro", "2023-02-14", 8200.00),
    (107, "Robert", "Marseille", "Particulier", "2023-05-01", 390.00),
    (108, "Richard", "Paris", "Pro", "2023-03-18", 1740.00),
    (109, "Moreau", "Lyon", "Collectivité", "2023-01-30", 2980.00),
    (110, "Simon", "Lille", "Particulier", "2023-06-12", 1075.00),
    (111, "Laurent", "Paris", "Pro", "2023-04-05", 15075.00),
    (112, "Michel", "Marseille", "Collectivité", "2023-02-28", 6650.00),
    (113, "Lefebvre", "Lyon", "Particulier", "2023-07-19", 460.00),
    (114, "Garcia", "Paris", "Pro", "2023-05-23", 1475.00),
    (115, "David", "Lille", "Collectivité", "2023-03-12", 4800.00)
]

# Création du DataFrame clients
df_clients = pd.DataFrame(clients_data, columns=[
    'client_id', 'nom', 'ville', 'segment', 'date_premiere_commande', 'ca_total_ht'
])

# =====================================================
# 3. DÉFINITION DES DONNÉES PRODUITS (catégories et prix)
# =====================================================

produits_data = [
    ("Ordinateur portable", "Électronique", 850.00),
    ("Écran 24 pouces", "Électronique", 180.00),
    ("Écran 27 pouces", "Électronique", 250.00),
    ("Serveur", "Électronique", 3200.00),
    ("Licence logiciel", "Logiciel", 150.00),
    ("Clavier", "Électronique", 45.00),
    ("Souris", "Électronique", 25.00),
    ("Casque audio", "Électronique", 65.00),
    ("Tableau blanc", "Mobilier", 120.00),
    ("Café", "Fourniture", 8.00),
    ("Papeterie", "Fourniture", 2.00),
    ("Formation Excel", "Service", 1200.00),
    ("Formation Power BI", "Service", 1000.00),
    ("Formation Python", "Service", 1500.00),
    ("Tableau de bord sur mesure", "Service", 2500.00)
]

# Création du DataFrame produits
df_produits = pd.DataFrame(produits_data, columns=[
    'produit', 'categorie', 'prix_unitaire_ht'
])

# =====================================================
# 4. AJOUT DES FEUILLES AU FICHIER EXCEL EXISTANT
# =====================================================

# Chemin du fichier Excel existant
chemin_excel = 'Projet10_dashboard_ventes_pme/data/ventes_brutes_PME.xlsx'

# Ouvrir le fichier existant avec openpyxl
with pd.ExcelWriter(chemin_excel, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    # Ajouter la feuille clients
    df_clients.to_excel(writer, sheet_name='clients', index=False)
    
    # Ajouter la feuille produits
    df_produits.to_excel(writer, sheet_name='produits', index=False)

# =====================================================
# 5. VÉRIFICATION ET RAPPORT
# =====================================================

# Lire les noms des feuilles pour vérifier
excel_file = pd.ExcelFile(chemin_excel)
feuilles = excel_file.sheet_names

print("=" * 60)
print("✅ FEUILLES AJOUTÉES AVEC SUCCÈS")
print("=" * 60)
print(f"📍 Fichier : {chemin_excel}")
print(f"📑 Feuilles présentes : {len(feuilles)}")
for i, feuille in enumerate(feuilles, 1):
    print(f"   {i}. {feuille}")

print("\n📊 Aperçu de la feuille 'clients' (5 premières lignes) :")
print(df_clients.head().to_string())

print("\n📊 Aperçu de la feuille 'produits' :")
print(df_produits.to_string())

print("\n" + "=" * 60)
print("🎯 LE FICHIER EXCEL EST MAINTENANT COMPLET :")
print("   - ventes_brutes (500 lignes)")
print("   - clients (15 références)")
print("   - produits (15 références)")
print("   - documentation (aide sur les colonnes)")
print("=" * 60)
print("\n📌 PROCHAIN ÉTAPE (Mercredi) :")
print("   P11 - Écrire les spécifications du script Python de nettoyage")
print("=" * 60)