import json
from sqlalchemy.orm import Session
from database import SessionLocal  # ta session de base de données
from models import Utilisateur     # ton modèle

# Charger le vecteur facial
with open("face_encoding_opencv.json", "r") as f:
    data = json.load(f)

# Convertir le vecteur en string JSON
face_encoding_str = json.dumps(data["encoding"])

# ID utilisateur à mettre à jour
target_user_id = 2

# Session DB
db: Session = SessionLocal()
try:
    # Récupérer l'utilisateur
    user = db.query(Utilisateur).filter(Utilisateur.id == target_user_id).first()

    if user:
        user.face_encoding = face_encoding_str
        db.commit()
        print(f"✅ Face encoding mis à jour pour l'utilisateur id={target_user_id}")
    else:
        print(f"❌ Utilisateur avec id={target_user_id} introuvable.")
finally:
    db.close()
