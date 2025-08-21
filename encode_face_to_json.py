import cv2
import json
import numpy as np
import os

def detect_face_and_save_vector(image_path, output_json="face_encoding_opencv.json"):
    if not os.path.exists(image_path):
        print(f"❌ Image introuvable : {image_path}")
        return

    # Charger le modèle Haar pour la détection de visage
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Lire l'image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Impossible de lire l'image.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Détection des visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        print("❌ Aucun visage détecté.")
        return

    # On prend le premier visage détecté
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]

    # Redimensionner à taille fixe pour avoir un vecteur cohérent
    face_resized = cv2.resize(face, (100, 100))  # (100x100x3)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

    # Conversion en vecteur 1D
    vector = face_rgb.flatten().astype(int).tolist()

    # Structure du JSON
    data = {
        "encoding": vector,
        "dimensions": len(vector),
        "source": os.path.basename(image_path),
        "method": "opencv_haar",
        "timestamp": str(np.datetime64("now"))
    }

    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Visage encodé et sauvegardé dans {output_json} ({len(vector)} valeurs)")

# ▶️ Exécution
if __name__ == "__main__":
    detect_face_and_save_vector("static/faces/face_2.jpg")
