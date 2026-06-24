# AutoVision Cabek

## Description

AutoVision Cabek est une application web basée sur l'Intelligence Artificielle permettant d'analyser automatiquement des images de véhicules.

L'application utilise un modèle YOLOv11 entraîné pour détecter les différentes pièces automobiles visibles sur une photo et fournir une estimation préliminaire du coût de réparation.

---

## Fonctionnalités

* Détection automatique des pièces automobiles
* Identification et localisation des composants du véhicule
* Analyse de plusieurs images simultanément
* Visualisation des résultats en temps réel
* Estimation du coût de réparation
* Génération d'un rapport d'analyse
* Export des résultats au format CSV

---

## Technologies utilisées

* Python
* Streamlit
* YOLOv11
* OpenCV
* NumPy
* Pandas
* Pillow

---

## Structure du projet

```text
AutoVision-Cabek/
│
├── app.py
├── best.pt
├── logo.png
├── requirements.txt
└── README.md
```

---

## Installation

Cloner le projet :

```bash
git clone https://github.com/asmaa1202/AutoVision-Cabek.git
```

Accéder au dossier :

```bash
cd AutoVision-Cabek
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer l'application :

```bash
streamlit run app.py
```

---

## Utilisation

1. Importer une ou plusieurs images du véhicule.
2. Lancer l'analyse automatique.
3. Consulter les pièces détectées.
4. Visualiser les résultats.
5. Télécharger le rapport CSV.

---

## Résultats

L'application fournit :

* Le nombre de pièces détectées
* Le niveau de confiance des prédictions
* Le temps d'analyse
* Le coût estimé des réparations
* Un rapport exportable

---

© 2026 AutoVision Cabek - Tous droits réservés.
