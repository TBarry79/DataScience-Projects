# Segmentez des clients d'un site e-commerce

## Compétences Mises en Œuvre
- Adapter les hyperparamètres d'un algorithme non supervisé afin de l'améliorer
- Évaluer les performances d’un modèle d'apprentissage non supervisé
- Transformer les variables pertinentes d'un modèle d'apprentissage non supervisé
- Mettre en place le modèle d'apprentissage non supervisé adapté au problème métier

## Description du Projet
Ce projet se concentre sur la segmentation des clients d'Olist, une entreprise brésilienne qui propose une solution de vente sur les marketplaces en ligne. L'objectif est de comprendre les différents types d'utilisateurs en analysant leur comportement d'achat et leurs données personnelles, pour optimiser les futures campagnes marketing.

## Objectif
- **Segmenter les clients** en utilisant des méthodes non supervisées.
- **Fournir une description claire** et exploitable de la segmentation.
- **Proposer un contrat de maintenance** basé sur l'analyse de la stabilité des segments dans le temps.

## Source des Données
Olist fournit une base de données anonymisée contenant des informations sur l'historique des commandes, les produits achetés, les commentaires de satisfaction, et la localisation des clients depuis janvier 2017. [Télécharger la base de données](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

## Analyse et Outils Utilisés
- **Clustering** : K-means et DBSCAN pour identifier des segments distincts.
- **Visualisation** : Matplotlib et Seaborn pour une présentation intuitive des résultats.
- **Prétraitement** : Standardisation des données et PCA pour la réduction de dimensionnalité.
- **Analyse** : Analyse approfondie des clusters pour générer des insights exploitables.

## Instructions Complémentaires
- Les données sont limitées, avec seulement 3 % des clients ayant réalisé plusieurs commandes.
- La segmentation doit différencier les bons et moins bons clients en termes de commandes et de satisfaction.
- Respecter la convention PEP8 dans le code fourni.

## Impact Attendu
L'analyse des comportements d'achat de 100 000 clients d'Olist a pour objectif d'optimiser les campagnes marketing, avec des cibles d'**augmentation de 30 % du taux d'engagement** et de **25 % des ventes** dans les six mois suivant l'implémentation.

## Contact
Pour toute question ou clarification, n'hésitez pas à me contacter.

[Revenir au guide principal ici](DataScience-Projects/README.md).

--- 
