# Classification Automatique des Biens de Consommation

## Compétences Mises en Œuvre
- Utiliser des techniques d’augmentation des données.
- Prétraiter des données texte pour obtenir un jeu de données exploitable.
- Représenter graphiquement des données à grandes dimensions.
- Prétraiter des données image pour obtenir un jeu de données exploitable.
- Mettre en œuvre des techniques de réduction de dimension.
- Définir la stratégie de collecte de données en recensant les API disponibles.
- Définir la stratégie d’élaboration d’un modèle d'apprentissage profond.
- Évaluer la performance des modèles d’apprentissage profond selon différents critères.

## Description du Projet
Ce projet consiste à étudier la faisabilité d'un moteur de classification automatique des articles pour la place de marché "Place de marché". L'objectif est d'automatiser l'attribution des catégories d'articles en utilisant à la fois les images et les descriptions textuelles des produits.

## Objectif
- **Réaliser une étude de faisabilité** en analysant les descriptions textuelles et les images des produits.
- **Mettre en place un moteur de classification automatique** démontrant la capacité à regrouper automatiquement des produits de même catégorie.

## Source des Données
L'entreprise "Place de marché" fournit une base de données contenant des photos et des descriptions d'articles. [Télécharger le jeu de données](lien_vers_le_jeu_de_donnees).

## Contraintes
- **Extraction des features texte** : Approches "bag-of-words" (comptage simple et Tf-idf), Word2Vec, BERT, USE.
- **Extraction des features image** : SIFT, ORB, SURF, CNN avec Transfer Learning.

## Résultats Attendus
- **Analyse visuelle** des clusters formés par les produits sur un graphique 2D.
- **Mesure de similarité** entre les catégories réelles et celles issues d'une segmentation en clusters.

## Analyse et Outils Utilisés
- **Textes** : Bag-of-Words, Word2Vec, BERT, et USE pour l'extraction et l'analyse des descriptions textuelles.
- **Images** : SIFT pour la détection des caractéristiques clés, Transfer Learning avec des réseaux de neurones pré-entraînés comme VGG16 ou ResNet pour l'extraction des informations visuelles.

## Impact Attendu
Ce moteur de classification a classé plus de 1050 produits avec une précision de 95%, réduisant de 50% le temps nécessaire à la catégorisation manuelle. Cela a permis d'améliorer la gestion des produits et les recommandations sur la plateforme.

## Instructions Complémentaires
- Utiliser l'exemple d'étude de faisabilité fourni comme point de départ.
- Mettre en œuvre la classification supervisée à partir des images dans la deuxième itération.

## Contact
Pour toute question ou clarification, n'hésitez pas à me contacter.

[Revenir au guide principal ici](DataScience-Projects/README.md).

--- 
