import pandas as pd
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime

# Import de VOTRE pipeline existant
from FoodCaloriePipeline.pipeline import FoodSegmentationClassificationPipeline

import os

def json_minute_aggregator(f, name=None):
    df = pd.read_json(f)
    df["dateTime"] = pd.to_datetime(df["dateTime"])

    # Add a day column without the second (just year, month, day)
    df["day"] = df["dateTime"].dt.date

    # Group by day and sum the calories
    df = df.groupby("day").agg({"value": "sum"}).reset_index()
    df.rename(columns={"day":"dateTime"}, inplace=True)

    if name is not None:
        df.rename(columns={"value": name}, inplace=True)

    return df

def json_minute_aggregator_data(data, name=None, aggregator="sum"):
    df = data.copy()
    df["dateTime"] = pd.to_datetime(df["dateTime"])

    # Add a day column without the second (just year, month, day)
    df["day"] = df["dateTime"].dt.date

    # Group by day and sum the calories
    df = df.groupby("day").agg({"value": aggregator}).reset_index()
    df.rename(columns={"day":"dateTime"}, inplace=True)

    if name is not None:
        df.rename(columns={"value": name}, inplace=True)

    return df



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
        
        res = []
        if len(detections) == 0:
            summary += "❌ Aucun aliment détecté. Essayez avec une autre image."
        else:
            summary += f"### {len(detections)} aliment(s) détecté(s) :\n\n"
            
            for i, detection in enumerate(detections, 1):
                summary += f"**{i}. {detection['class_name']}**\n"
                summary += f"   - 🎯 Confiance : {detection['confidence']:.1%}\n"
                summary += f"   - ⚖️ Poids estimé : {detection['weight_grams']:.0f}g\n"
                summary += f"   - 🔥 Calories : {detection['estimated_calories']:.0f} kcal\n\n"
                res.append({
                    "class_name": detection['class_name'],
                    "confidence": detection['confidence'],
                    "weight_grams": detection['weight_grams'],
                    "estimated_calories": detection['estimated_calories']
                })

        return annotated_image, summary, res

    except Exception as e:
        print(f"Erreur : {e}")
        return None, f"❌ Erreur lors de l'analyse : {str(e)}", []
    
def process_food_images(folder):
    """
    Process food images from a folder.
    """
    image_folder = f"data/{folder}/food-images"
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    df = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        with open(image_path, "rb") as img_file:
            img = Image.open(img_file)
            exifdata = img.getexif()
            for tag_id, value in exifdata.items():
                tag = TAGS.get(tag_id, tag_id)

                if tag in ("DateTime", "DateTimeOriginal", "DateTimeDigitized"):
                    dt = value
        
            annotated_image, summary, res = analyze_food(img)
            df.append({
                "date_time": dt,
                "results": res
            })

    df_food = pd.DataFrame(df)
    df_food["dateTime"] = df_food["date_time"].apply(lambda x: datetime.strptime(x, "%Y:%m:%d %H:%M:%S") )
    df_food["dateTime"] = df_food["dateTime"].dt.date
    df_food = df_food.sort_values("dateTime")
    df_food.drop(columns=["date_time"], inplace=True)
    df_food["calories_in"] = df_food["results"].apply(lambda x: sum(item['estimated_calories'] for item in x if 'estimated_calories' in item))
    df_food["number_of_missing_meals"] = df_food["results"].apply(lambda x: 1 if len(x) == 0 else 0)
    df_food.drop(columns=["results"], inplace=True)
    df_food = df_food.groupby("dateTime").agg({"calories_in": "sum", "number_of_missing_meals": "sum"}).reset_index()

    return df_food

def process_one_player(folder):
    """
    Process one player's data from a folder.
    """
    list_df = []
    for metric in ["calories", "distance", "steps"]:
        f = f"data/{folder}/fitbit/{metric}.json"
        if os.path.exists(f):
            df = json_minute_aggregator(f, metric)
            if metric == "distance":
                df["distance"] = df["distance"] / 100000.0
            list_df.append(df)
        else:
            print(f"File not found: {f}")
            list_df.append(pd.DataFrame(columns=["dateTime", metric]))

    df_aggregated = pd.merge(list_df[0], list_df[1], on="dateTime", how="left")
    df_aggregated = pd.merge(df_aggregated, list_df[2], on="dateTime", how="left")

    # heart rate
    f_hr = f"data/{folder}/fitbit/heart_rate.json"
    if os.path.exists(f_hr):
        df_hr = pd.read_json(f_hr)
        df_hr["value"] = df_hr["value"].apply(lambda x: x["bpm"])
        df_hr = json_minute_aggregator_data(df_hr, "heart_rate", aggregator="mean")
        df_aggregated = pd.merge(df_aggregated, df_hr, on="dateTime", how="left")

    # Resting heart rate
    f_rhr = f"data/{folder}/fitbit/resting_heart_rate.json"
    if os.path.exists(f_rhr):
        df_rhr = pd.read_json(f_rhr)
        df_rhr["dateTime"] = pd.to_datetime(df_rhr["dateTime"])
        df_rhr["dateTime"] = df_rhr["dateTime"].dt.date
        df_rhr["value"] = df_rhr["value"].apply(lambda x: x["value"])
        df_rhr.rename(columns={"value": "resting_heart_rate"}, inplace=True)
        df_aggregated = pd.merge(df_aggregated, df_rhr, on="dateTime", how="left")

    # time_in_heart_rate_zones
    f_zones = f"data/{folder}/fitbit/time_in_heart_rate_zones.json"
    if os.path.exists(f_zones):
        df_time_in_zones = pd.read_json(f_zones)
        df_time_in_zones["dateTime"] = pd.to_datetime(df_time_in_zones["dateTime"])
        df_time_in_zones["dateTime"] = df_time_in_zones["dateTime"].dt.date
        for key in df_time_in_zones.iloc[0]["value"].get("valuesInZones").keys():
            df_time_in_zones[key] = df_time_in_zones["value"].apply(lambda x: x["valuesInZones"][key])
        df_aggregated = pd.merge(df_aggregated, df_time_in_zones, on="dateTime", how="left")

    # sleep
    f_sleep = f"data/{folder}/fitbit/sleep.json"
    if os.path.exists(f_sleep):
        df_sleep = pd.read_json(f_sleep)
        for key in ["restless", "wake", "rem", "light", "deep"]:
            df_sleep[f"sleep_period_{key}_duration"] = df_sleep["levels"].apply(lambda x: x["summary"][key].get("minutes") if key in x["summary"] else 0)
        df_sleep["dateTime"] = pd.to_datetime(df_sleep["dateOfSleep"])
        df_sleep["dateTime"] = df_sleep["dateTime"].dt.date
        df_sleep = df_sleep[["dateTime", "sleep_period_deep_duration", "sleep_period_light_duration", "sleep_period_rem_duration", "sleep_period_wake_duration", "minutesAsleep", "minutesAwake", "timeInBed"]]
        df_aggregated = pd.merge(df_aggregated, df_sleep, on="dateTime", how="left")

    # sleep score
    f_sleep_score = f"data/{folder}/fitbit/sleep_score.csv"
    if os.path.exists(f_sleep_score):
        df_sleep_score = pd.read_csv(f_sleep_score)
        df_sleep_score["dateTime"] = pd.to_datetime(df_sleep_score["timestamp"])
        df_sleep_score["dateTime"] = df_sleep_score["dateTime"].dt.date
        df_sleep_score.drop(columns=["timestamp", "sleep_log_entry_id"], inplace=True)
        for col in df_sleep_score.columns:
            if col != "dateTime":
                df_sleep_score.rename(columns={col: f"sleep_score_{col}"}, inplace=True)
        df_aggregated = pd.merge(df_aggregated, df_sleep_score, on="dateTime", how="left")

    # exercise data
    f_exercise = f"data/{folder}/fitbit/exercise.json"
    if os.path.exists(f_exercise):
        df_exercise = pd.read_json(f_exercise)
        df_exercise["dateTime"] = pd.to_datetime(df_exercise["startTime"])
        df_exercise["dateTime"] = df_exercise["dateTime"].dt.date
        for i, key in enumerate(["sedentary", "lightly", "fairly", "very"]):
            df_exercise[f"duration_at_activity_level_{key}"] = df_exercise["activityLevel"].apply(lambda x: x[i].get("minutes"))
        df_exercise = df_exercise[["dateTime", "duration_at_activity_level_sedentary", "duration_at_activity_level_lightly", "duration_at_activity_level_fairly", "duration_at_activity_level_very"]]
        df_exercise = df_exercise.groupby("dateTime").sum().reset_index()
        df_aggregated = pd.merge(df_aggregated, df_exercise, on="dateTime", how="left")

    # wellness
    f_wellness = f"data/{folder}/pmsys/wellness.csv"
    if os.path.exists(f_wellness):
        df_wellness = pd.read_csv(f_wellness)
        df_wellness["dateTime"] = pd.to_datetime(df_wellness["effective_time_frame"])
        df_wellness["dateTime"] = df_wellness["dateTime"].dt.date
        df_wellness.drop(columns=["effective_time_frame"], inplace=True)
        df_aggregated = pd.merge(df_aggregated, df_wellness, on="dateTime", how="left")

    # srpe
    f_srpe = f"data/{folder}/pmsys/srpe.csv"
    if os.path.exists(f_srpe):
        df_srpe = pd.read_csv(f_srpe)
        df_srpe["dateTime"] = pd.to_datetime(df_srpe["end_date_time"])
        df_srpe["dateTime"] = df_srpe["dateTime"].dt.date
        df_srpe.drop(columns=["end_date_time"], inplace=True)
        df_aggregated = pd.merge(df_aggregated, df_srpe, on="dateTime", how="left")

    # reporting
    f_reporting = f"data/{folder}/googledocs/reporting.csv"
    if os.path.exists(f_reporting):
        df_reporting = pd.read_csv(f_reporting)
        df_reporting.rename(columns={"date": "dateTime"}, inplace=True)
        df_reporting["dateTime"] = df_reporting["dateTime"].apply(lambda x: datetime.strptime(x, "%d/%m/%Y") )
        df_reporting["dateTime"] = df_reporting["dateTime"].dt.date
        df_reporting.drop(columns=["timestamp"], inplace=True)
        df_aggregated = pd.merge(df_aggregated, df_reporting, on="dateTime", how="left")

    # injury
    f_injury = f"data/{folder}/pmsys/injury.csv"
    if os.path.exists(f_injury):
        df_injury = pd.read_csv(f_injury)
        df_injury["dateTime"] = pd.to_datetime(df_injury["effective_time_frame"])
        df_injury["dateTime"] = df_injury["dateTime"].dt.date
        df_injury.drop(columns=["effective_time_frame"], inplace=True)
        df_aggregated = pd.merge(df_aggregated, df_injury, on="dateTime", how="left")

    # food images
    df_food = process_food_images(folder)
    df_aggregated = pd.merge(df_aggregated, df_food, on="dateTime", how="left")

    df_aggregated["Player"] = folder
    return df_aggregated

"""
Fonction principale pour exécuter le pipeline.
"""
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
folders = ["p01", "p03", "p05"]  # Remplacez par le nom du dossier de votre joueur
df = []
print("Traitement des données...")
for folder in folders:
    print(f"Traitement du joueur {folder}...")
    df_aggregated = process_one_player(folder)
    df.append(df_aggregated)
print("✅ Traitement terminé!")

df_aggregated = pd.concat(df, ignore_index=True)
df_aggregated.to_csv("test_dataset.csv")
# Afficher les 10 premières lignes du DataFrame agrégé
print(df_aggregated.head(10))