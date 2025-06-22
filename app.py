import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import io
import base64
import sys

# Configuration de la page
st.set_page_config(
    page_title="Diagnostic Cancer du Poumon",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé moderne et professionnel
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .stApp {
        background: transparent;
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stMarkdown h1 {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 2rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stMarkdown h2 {
        color: #2c3e50;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
    }
    
    .stMarkdown h3 {
        color: #34495e;
        font-weight: 500;
        margin: 1.5rem 0 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .sidebar .sidebar-content .stMarkdown {
        color: white;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        text-align: center;
    }
    
    .example-section {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
    }
    
    .results-section {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stRadio > div {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    .stAlert {
        border-radius: 15px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    }
    
    .danger-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .warning-section {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        color: #2d3436;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0;
        border-left: 5px solid #e17055;
    }
    
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 2px dashed rgba(255, 255, 255, 0.5);
        border-radius: 15px;
        padding: 2rem;
    }
    
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Animation pour les cartes */
    .card-hover {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card-hover:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .stMarkdown h1 {
            font-size: 2rem;
        }
        
        .main-container {
            margin: 0.5rem;
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Augmenter la limite de récursion (temporaire)
original_recursion_limit = sys.getrecursionlimit()
sys.setrecursionlimit(10000)

# Chargement du modèle avec gestion d'erreur améliorée
@st.cache_resource
def load_model():
    """
    Charge le modèle avec plusieurs méthodes de fallback pour éviter les erreurs de récursion
    """
    try:
        
        
        
        
        # Chemins possibles pour le modèle
        model_paths = [
            'trained_lung_cancer_model.h5',
            './trained_lung_cancer_model.h5',
            os.path.join(os.getcwd(), 'trained_lung_cancer_model.h5'),
            os.path.join(os.path.dirname(__file__), 'trained_lung_cancer_model.h5')
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                
                break
        
        if not model_path:
            st.error("❌ Modèle non trouvé dans les chemins suivants:")
            for path in model_paths:
                st.write(f"  - {path}")
            return None
        
        
        
        # Méthode 1: Chargement standard avec compile=False
        try:
            st.info("🔄 Tentative de chargement standard...")
            model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects=None
            )
            st.success("✅ Modèle chargé avec succès (méthode standard)!")
            return model
            
        except (RecursionError, RuntimeError) as e:
            st.warning(f"⚠️ Erreur de récursion avec méthode standard: {str(e)[:100]}...")
            
            # Méthode 2: Chargement avec options spécifiques
            try:
                st.info("🔄 Tentative de chargement avec options personnalisées...")
                
                # Réinitialiser le backend Keras
                tf.keras.backend.clear_session()
                
                # Charger avec des options spécifiques
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False,
                    custom_objects=None,
                    safe_mode=False  # Pour TensorFlow 2.11+
                )
                st.success("✅ Modèle chargé avec succès (méthode personnalisée)!")
                return model
                
            except Exception as e2:
                st.warning(f"⚠️ Erreur avec méthode personnalisée: {str(e2)[:100]}...")
                
                # Méthode 3: Chargement des poids séparément
                try:
                    st.info("🔄 Tentative de reconstruction du modèle...")
                    
                    # Créer un modèle simple pour le test
                    model = tf.keras.Sequential([
                        tf.keras.layers.Input(shape=(224, 224, 3)),
                        tf.keras.layers.Conv2D(32, 3, activation='relu'),
                        tf.keras.layers.GlobalAveragePooling2D(),
                        tf.keras.layers.Dense(128, activation='relu'),
                        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
                    ])
                    
                    # Essayer de charger les poids si possible
                    try:
                        model.load_weights(model_path)
                        st.success("✅ Modèle reconstruit et poids chargés!")
                        return model
                    except:
                        st.warning("⚠️ Impossible de charger les poids, utilisation d'un modèle par défaut")
                        return model
                        
                except Exception as e3:
                    st.error(f"❌ Erreur lors de la reconstruction: {str(e3)[:100]}...")
                    
                    # Méthode 4: Modèle de fallback
                    st.info("🔄 Création d'un modèle de démonstration...")
                    model = create_fallback_model()
                    st.warning("⚠️ Utilisation d'un modèle de démonstration (non entraîné)")
                    return model
        
    except ImportError as e:
        st.error(f"❌ Erreur d'importation: {str(e)}")
        st.error("Veuillez installer les dépendances: pip install tensorflow keras")
        return None
    except Exception as e:
        st.error(f"❌ Erreur générale: {str(e)}")
        return None
    finally:
        # Restaurer la limite de récursion originale
        sys.setrecursionlimit(original_recursion_limit)

def create_fallback_model():
    """
    Crée un modèle de démonstration simple pour les tests
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
    ])
    
    # Compiler le modèle
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Prétraitement de l'image amélioré
def preprocess_image(image_input):
    try:
        if isinstance(image_input, str):
            # Si c'est un chemin de fichier
            img = Image.open(image_input)
        else:
            # Si c'est un objet de fichier uploadé
            img = Image.open(image_input)
        
        # Convertir en RGB si nécessaire
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionner
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Normaliser les pixels
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Ajouter la dimension batch
        img_batch = np.expand_dims(img_array, axis=0)
        
        return img_batch, img
        
    except Exception as e:
        st.error(f"❌ Erreur de traitement d'image: {str(e)}")
        return None, None

# Fonction pour créer des images d'exemple (placeholder)
def create_example_image(class_name, color):
    """Crée une image d'exemple avec du texte"""
    img = Image.new('RGB', (224, 224), color=color)
    return img

# Fonction pour afficher les résultats avec style
def display_results(predicted_class, confidence, predictions, class_names):
    st.markdown("## 📊 Résultats de l'analyse")
    
    # Affichage du résultat principal
    if predicted_class == "Bénin":
        st.markdown(f"""
        <div class="success-card">
            <h3>✅ Résultat: {predicted_class}</h3>
            <p>Tissu classifié comme bénin</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="danger-card">
            <h3>⚠️ Résultat: {predicted_class}</h3>
            <p>Attention: Tissu potentiellement cancéreux détecté</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Métrique de confiance
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Niveau de confiance</h4>
            <h2>{confidence*100:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphique des probabilités
    st.markdown("### 📈 Distribution des probabilités")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    bars = ax.bar(class_names.values(), predictions[0], color=colors, alpha=0.8)
    
    # Ajouter les valeurs sur les barres
    for bar, prob in zip(bars, predictions[0]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilité", fontsize=12, fontweight='bold')
    ax.set_xlabel("Classes", fontsize=12, fontweight='bold')
    ax.set_title("Probabilités de classification", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Styliser le graphique
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    st.pyplot(fig)

# Interface principale
def main():
    # Conteneur principal
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.title("🫁 Diagnostic Assisté du Cancer du Poumon")
    
    # Sidebar avec informations
    with st.sidebar:
        st.markdown("## ℹ️ À propos")
        st.markdown("""
        Cette application utilise l'intelligence artificielle pour analyser les images histopathologiques de tissus pulmonaires.
        
        
        **Classifications possibles :**
        - 🔴 **Adénocarcinoma** 
        - 🟢 **Bénin** 
        - 🔵 **squamous_cell_carcinoma** 
        """)
        
        st.markdown("---")
        st.markdown("## ⚙️ Paramètres")
        show_details = st.checkbox("Afficher les détails techniques", value=False)
        confidence_threshold = st.slider("Seuil de confiance", 0.5, 1.0, 0.8, 0.05)
    
    # Chargement du modèle
    model = load_model()
    
    if model is None:
        st.error("Impossible de charger le modèle. Vérifiez les messages ci-dessus pour plus de détails.")
        st.stop()
    
    # Section de téléchargement
    st.markdown("""
    <div class="upload-section">
        <h3>📤 Téléverser une image histopathologique</h3>
        <p>Formats acceptés: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choisissez une image...", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    # Section des exemples
    st.markdown("""
    <div class="example-section">
        <h3>🖼️ Ou utilisez un exemple de démonstration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    example_choice = st.radio(
        "Sélectionnez un exemple :",
        ["Adénocarcinome", "Bénin", "squamous_cell_carcinoma"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
# Traitement et affichage
# Configuration des exemples
EXAMPLE_IMAGES = {
    "Adénocarcinome": "C:\\Users\\HP\\Documents\\dockerFom\\machinelearning\\data_scrap\\DeepLearning_LungCancer\\adenocarcinoma_example.jpg",
    "Bénin": "C:\\Users\\HP\\Documents\\dockerFom\\machinelearning\\data_scrap\\DeepLearning_LungCancer\\benign_example.jpg", 
    "squamous_cell_carcinoma": "C:\\Users\\HP\\Documents\\dockerFom\\machinelearning\\data_scrap\\DeepLearning_LungCancer\\squamous_example.jpg"
}
# Vérification que les fichiers existent
for name, path in EXAMPLE_IMAGES.items():
    if not os.path.exists(path):
        st.warning(f"Fichier exemple manquant: {path}")

def check_example_images():
    for name, path in EXAMPLE_IMAGES.items():
        if not os.path.exists(path):
            st.warning(f"Fichier exemple manquant: {path}")

def main():
    # Conteneur principal
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.title("🫁 Diagnostic Assisté du Cancer du Poumon")
    
    # Sidebar avec informations
    with st.sidebar:
        st.markdown("## ℹ️ À propos")
        st.markdown("""
        Cette application utilise l'intelligence artificielle pour analyser les images histopathologiques de tissus pulmonaires.
        
        **Classifications possibles :**
        - 🔴 **Adénocarcinoma** 
        - 🟢 **Bénin**
        - 🔵 **squamous_cell_carcinoma** 
        """)
        
        st.markdown("---")
        st.markdown("## ⚙️ Paramètres")
        show_details = st.checkbox("Afficher les détails techniques", value=False)
        confidence_threshold = st.slider("Seuil de confiance", 0.5, 1.0, 0.8, 0.05)
    
    # Chargement du modèle
    model = load_model()
    
    if model is None:
        st.error("Impossible de charger le modèle. Vérifiez les messages ci-dessus pour plus de détails.")
        st.stop()
    
    # Section de téléchargement
    st.markdown("""
    <div class="upload-section">
        <h3>📤 Téléverser une image histopathologique</h3>
        <p>Formats acceptés: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choisissez une image...", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    # Section des exemples
    st.markdown("""
    <div class="example-section">
        <h3>🖼️ Ou utilisez un exemple de démonstration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    example_choice = st.radio(
        "Sélectionnez un exemple :",
        ["Adénocarcinome", "Bénin", "squamous_cell_carcinoma"],
        horizontal=True,
        label_visibility="collapsed"
    )

    check_example_images()
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🖼️ Image à analyser")
        img_to_process = None
        
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Image téléversée", use_container_width=True)
            img_to_process = uploaded_file
        else:
            # Charger l'image d'exemple
            example_path = EXAMPLE_IMAGES[example_choice]
            if os.path.exists(example_path):
                example_img = Image.open(example_path)
                st.image(example_img, caption=f"Exemple: {example_choice}", use_container_width=True)
                img_to_process = example_path
            else:
                # Si l'image d'exemple n'existe pas, créer une image de démonstration colorée
                example_colors = {
                    "Adénocarcinome": "#e74c3c",
                    "Bénin": "#2ecc71",
                    "squamous_cell_carcinoma": "#3498db"
                }
                example_img = create_example_image(example_choice, example_colors[example_choice])
                st.image(example_img, caption=f"Exemple: {example_choice}", use_container_width=True)
                # Convertir l'image en bytes pour le traitement
                img_bytes = io.BytesIO()
                example_img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                img_to_process = img_bytes
    
    with col2:
        st.markdown("### 🔍 Analyse")
        
        if img_to_process and model:
            if st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyse en cours..."):
                    try:
                        # Prétraitement
                        img_array, processed_img = preprocess_image(img_to_process)
                        
                        if img_array is not None:
                            # Prédiction avec gestion d'erreur
                            predictions = model.predict(img_array, verbose=0)
                            class_idx = np.argmax(predictions[0])
                            confidence = np.max(predictions[0])
                            
                            # Mapping des classes
                            class_names = {
                                0: "Adénocarcinoma",
                                1: "Bénin", 
                                2: "squamous_cell_carcinoma"
                            }
                            predicted_class = class_names[class_idx]
                            
                            # Affichage des résultats
                            display_results(predicted_class, confidence, predictions, class_names)
                            
                            # Vérification du seuil de confiance
                            if confidence < confidence_threshold:
                                st.warning(f"⚠️ Confiance faible ({confidence*100:.1f}% < {confidence_threshold*100:.0f}%). Résultat à interpréter avec précaution.")
                            
                            # Détails techniques
                            if show_details:
                                st.markdown("### 🔍 Détails techniques")
                                with st.expander("Voir les détails"):
                                    st.json({
                                        "Prédictions brutes": [f"{prob:.6f}" for prob in predictions[0]],
                                        "Classe prédite": predicted_class,
                                        "Indice de classe": int(class_idx),
                                        "Confiance": f"{confidence:.6f}",
                                        "Dimensions image": list(img_array.shape),
                                        "Seuil de confiance": confidence_threshold,
                                        "Limite de récursion": sys.getrecursionlimit()
                                    })
                    
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
                        st.info("💡 Essayez de redémarrer l'application ou de réduire la taille de l'image.")
        else:
            st.info("📋 Téléversez une image ou sélectionnez un exemple pour commencer l'analyse.")
    
    # Avertissement médical
    st.markdown("""
    <div class="warning-section">
        <h4>⚠️ Avertissement médical important</h4>
        <p>
        <strong>Ce système est uniquement un outil d'aide au diagnostic et ne remplace EN AUCUN CAS l'expertise d'un professionnel de santé qualifié.</strong>
        </p>
        <ul>
            <li>Les résultats doivent être interprétés par un médecin spécialisé</li>
            <li>Ne prenez aucune décision médicale basée uniquement sur ces résultats</li>
            <li>Consultez toujours un professionnel de santé pour un diagnostic définitif</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()