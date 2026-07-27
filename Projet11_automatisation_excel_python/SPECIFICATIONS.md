# Spécifications techniques - Script de nettoyage Excel

## Objectif général
Développer un script Python qui nettoie automatiquement un fichier Excel brut de ventes et génère un fichier propre prêt à l'analyse.

## Périmètre
Le script doit traiter le fichier `ventes_brutes_PME.xlsx` (500 lignes) créé dans le projet P10.

---

## 1. Lecture du fichier source

### 1.1 Fichier d'entrée
| Propriété | Valeur |
|-----------|--------|
| Chemin | `data/ventes_brutes_PME.xlsx` |
| Feuille | `ventes_brutes` |

### 1.2 Vérification préalable
- [ ] Vérifier que le fichier existe
- [ ] Vérifier que la feuille `ventes_brutes` existe
- [ ] Vérifier que les colonnes attendues sont présentes

---

## 2. Règles de nettoyage

### 2.1 Suppression des lignes inutiles
| Règle | Action | Pourquoi |
|-------|--------|----------|
| `montant_total_ht` est vide (NaN) | Supprimer la ligne | Donnée manquante exploitable |
| `produit` est vide | Supprimer la ligne | Impossible d'analyser |
| `client_id` est vide | Supprimer la ligne | Impossible de rattacher au client |

### 2.2 Correction des valeurs aberrantes
| Problème | Action |
|----------|--------|
| `montant_total_ht` < 0 | Remplacer par `0` (ou supprimer après alerte) |
| `quantite` < 1 | Remplacer par `1` |
| `prix_unitaire_ht` < 0 | Remplacer par la moyenne des prix du même produit |

### 2.3 Standardisation des formats
| Colonne | Problème possible | Action |
|---------|-------------------|--------|
| `date_vente` | Format texte | Convertir en datetime (YYYY-MM-DD) |
| `statut` | Variations (Payé, PAYE, paye) | Uniformiser : 'Payé', 'En attente', 'Annulé' |
| `client_nom` | Espaces en trop | Nettoyer avec `.strip()` |

### 2.4 Suppression des doublons
- [ ] Identifier les doublons sur `vente_id`
- [ ] Supprimer les lignes strictement identiques

---

## 3. Enrichissement des données (optionnel)

### 3.1 Ajout de colonnes
| Nouvelle colonne | Calcul |
|------------------|--------|
| `annee` | Extraire l'année de `date_vente` |
| `mois` | Extraire le mois |
| `trimestre` | Calculer à partir du mois |
| `jour_semaine` | Lundi=1 ... Dimanche=7 |

### 3.2 Jointure avec la table produits
- Ajouter la colonne `categorie` en se basant sur `produit`
- Source : feuille `produits` du même fichier

---

## 4. Contrôles qualité (validation)

### 4.1 Vérifications après nettoyage
| Contrôle | Attendu |
|----------|---------|
| Nombre de lignes après nettoyage | > 450 (les aberrantes supprimées/corrigées) |
| `montant_total_ht` minimum | >= 0 |
| `quantite` minimum | >= 1 |
| Aucune ligne avec `statut` inconnu | 'Payé', 'En attente' ou 'Annulé' |

### 4.2 Génération d'un rapport de qualité
Le script doit afficher :