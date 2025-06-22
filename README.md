# Diagnostic Cancer du Poumon – Deep Learning Streamlit App

Cette application Streamlit permet d'analyser des images histopathologiques de tissus pulmonaires pour assister au diagnostic du cancer du poumon à l'aide d'un modèle de deep learning.

## Fonctionnalités

- Téléversement d'images histopathologiques (JPG, JPEG, PNG)
- Analyse automatique par un modèle TensorFlow/Keras
- Prédiction de la classe : 
  - Adénocarcinoma
  - Bénin
  - squamous_cell_carcinoma
- Affichage du niveau de confiance et des probabilités
- Visualisation graphique des résultats
- Exemples d'images intégrés
- Interface moderne et responsive
- Avertissement médical intégré

## Pile technologique

- Python
- Streamlit
- TensorFlow / Keras
- NumPy
- Pillow (PIL)
- Matplotlib

## Prise en main

1. **Installer les dépendances :**

   ```sh
   pip install streamlit tensorflow numpy pillow matplotlib
   ```

2. **Placer le modèle entraîné :**

   Placez le fichier `trained_lung_cancer_model.h5` dans le dossier [`DeepLearning_LungCancer`](app.py).

3. **(Optionnel) Ajouter des images d'exemple :**

   Placez les fichiers suivants dans le dossier :
   - `adenocarcinoma_example.jpg`
   - `benign_example.jpg`
   - `squamous_example.jpg`

4. **Lancer l'application :**

   ```sh
   streamlit run app.py
   ```

## Structure des fichiers

- [`app.py`](app.py) : Application principale Streamlit
- `trained_lung_cancer_model.h5` : Modèle de deep learning (à fournir)
- `adenocarcinoma_example.jpg`, `benign_example.jpg`, `squamous_example.jpg` : Images d'exemple (optionnel)

## Avertissement

> **Cette application est un outil d'aide au diagnostic et ne remplace en aucun cas l'avis d'un professionnel de santé qualifié. Les résultats doivent toujours être interprétés par un médecin.**

---

*Développé pour l'analyse assistée par IA des images histopathologiques pulmonaires.*