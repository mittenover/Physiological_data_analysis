# app.py
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import time

# Import de VOTRE pipeline existant
from pipeline import FoodSegmentationClassificationPipeline

# Configuration des chemins
YOLO_WEIGHTS = "FoodCaloriePipeline/model_weights/best.pt"
MOBILENET_WEIGHTS = "FoodCaloriePipeline/model_weights/mobilenetv2_best.pth"
CLASS_NAMES_FILE = "FoodCaloriePipeline/assets/food_labels.txt"
WEIGHT_CSV = "FoodCaloriePipeline/assets/average_weight.csv"
CALORIE_CSV = "FoodCaloriePipeline/assets/calorie_per_100g.csv"

# Initialiser le pipeline (exactement comme dans votre main.py)
print("Initialisation du pipeline...")
pipeline = FoodSegmentationClassificationPipeline(
    yolo_weights=YOLO_WEIGHTS,
    mobilenet_weights=MOBILENET_WEIGHTS,
    class_names_path=CLASS_NAMES_FILE,
    weight_csv=WEIGHT_CSV,
    calorie_csv=CALORIE_CSV
)
print("✅ Pipeline prêt!")

def analyze_food(image):
    """Fonction principale pour Gradio"""
    if image is None:
        return None, "Veuillez uploader une image"
    
    try:
        # Convertir PIL en numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convertir RGB en BGR pour OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Utiliser VOTRE pipeline
        detections = pipeline.process_image(image_bgr)
        
        # Créer l'image annotée
        annotated_image = image.copy()
        
        # Dessiner les détections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Rectangle
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Texte avec fond
            label = f"{detection['class_name']} ({detection['confidence']:.0%})"
            calories = f"{detection['estimated_calories']:.0f} kcal"
            
            # Fond pour le texte
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(annotated_image, (x1, y1-h-10), (x1+w, y1), (0, 255, 0), -1)
            cv2.putText(annotated_image, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Calories
            cv2.putText(annotated_image, calories, (x1, y2+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Créer le résumé
        total_calories = sum(d['estimated_calories'] for d in detections)
        
        summary = f"## 📊 Résultats de l'analyse\n\n"
        summary += f"### Total estimé : **{total_calories:.0f} kcal**\n\n"
        
        if len(detections) == 0:
            summary += "❌ Aucun aliment détecté. Essayez avec une autre image."
        else:
            summary += f"### {len(detections)} aliment(s) détecté(s) :\n\n"
            
            for i, detection in enumerate(detections, 1):
                summary += f"**{i}. {detection['class_name']}**\n"
                summary += f"   - 🎯 Confiance : {detection['confidence']:.1%}\n"
                summary += f"   - ⚖️ Poids estimé : {detection['weight_grams']:.0f}g\n"
                summary += f"   - 🔥 Calories : {detection['estimated_calories']:.0f} kcal\n\n"
        
        return annotated_image, summary
        
    except Exception as e:
        print(f"Erreur : {e}")
        return None, f"❌ Erreur lors de l'analyse : {str(e)}"

# Créer l'interface Gradio
with gr.Blocks(title="Food Detection & Calorie Estimation") as demo:
    gr.Markdown("""
    # 🍽️ Food Detection & Calorie Estimation
    
    Uploadez une photo de votre plat pour détecter automatiquement les aliments et estimer les calories.
    
    ### 🎯 Comment utiliser :
    1. Cliquez sur "Upload Image" ou glissez-déposez une photo
    2. Attendez l'analyse (quelques secondes)
    3. Visualisez les résultats !
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="📸 Votre photo",
                type="pil"
            )
            analyze_btn = gr.Button("🔍 Analyser", variant="primary", size="lg")
            
        with gr.Column():
            output_image = gr.Image(
                label="📊 Résultats visuels"
            )
            
    results_text = gr.Markdown(label="📝 Détails")
    
    # Exemples (optionnel - commentez si vous n'avez pas d'images d'exemple)
    # gr.Examples(
    #     examples=[["exemple1.jpg"], ["exemple2.jpg"]],
    #     inputs=input_image
    # )
    
    # Connecter le bouton
    analyze_btn.click(
        fn=analyze_food,
        inputs=input_image,
        outputs=[output_image, results_text]
    )
    
    gr.Markdown("""
    ---
    ### ℹ️ À propos
    - **Modèles** : YOLOv8 (segmentation) + MobileNetV2 (classification)
    - **Classes** : 98 types d'aliments
    - **Estimation** : Basée sur le poids moyen de chaque aliment
    """)

# Lancer l'application
if __name__ == "__main__":
    demo.launch()