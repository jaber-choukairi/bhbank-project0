from fastapi import FastAPI, Request, Form, Depends, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Utilisateur, Application
from pydantic import BaseModel
from utils.email import send_reset_email
import secrets
import base64
import json
from PIL import Image
import numpy as np
import cv2
import re
from io import BytesIO
import os
import face_recognition
import tempfile
from typing import Optional


from encode_face_to_json import detect_face_and_save_vector

app = FastAPI()
USE_FACE_AUTH =1 # 1 = mot de passe + visage, 0 = mot de passe seulement
# Fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates HTML
templates = Jinja2Templates(directory="templates")



# Connexion DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Détection faciale
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -----------------------------
# Fonctions utilitaires
# -----------------------------
def extract_face_encoding(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("❌ Aucun visage détecté")
        return None

    x, y, w, h = faces[0]
    face = image_np[y:y+h, x:x+w]

    # Redimensionner pour avoir un vecteur cohérent
    face_resized = cv2.resize(face, (100, 100))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    vector = face_rgb.flatten()

    return vector

def compare_faces(face1_vec, face2_vec, threshold=15000.0):  # <- temporairement élevé

    # Normaliser les vecteurs
    face1 = np.array(face1_vec, dtype=np.float32)
    face2 = np.array(face2_vec, dtype=np.float32)

    if face1.shape != face2.shape:
        return False

    distance = np.linalg.norm(face1 - face2)
    print(f"🔍 Distance entre visages: {distance:.4f}")
    return distance < threshold

def decode_base64_image(base64_str):
    try:
        if "," not in base64_str:
            raise ValueError("Base64 string mal formée (pas de virgule)")

        header, encoded = base64_str.split(",", 1)
        img_data = base64.b64decode(encoded)
        img = Image.open(BytesIO(img_data)).convert("RGB")

        # 💡 Convertir proprement en array NumPy
        img_np = np.asarray(img, dtype=np.uint8)
        img_np = np.ascontiguousarray(img_np)

        print("🧪 decode >> shape:", img_np.shape, "dtype:", img_np.dtype)
        return img_np

    except Exception as e:
        print(f"❌ Erreur décodage image: {e}")
        return None

# -----------------------------
# Routes HTML
# -----------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
def get_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login", response_class=HTMLResponse)
def post_login(
        request: Request,
        email: str = Form(...),
        password: str = Form(...),
        db: Session = Depends(get_db)
):
    user = db.query(Utilisateur).filter_by(email=email).first()
    if user and password == user.password:
        return RedirectResponse(url="/dashboard", status_code=302)

    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": "Email ou mot de passe incorrect ❌"
    })

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/forgot-password", response_class=HTMLResponse)
def forgot_password_form(request: Request):
    return templates.TemplateResponse("forgot_password.html", {"request": request})

@app.post("/forgot-password", response_class=HTMLResponse)
def forgot_password(
        request: Request,
        email: str = Form(...),
        db: Session = Depends(get_db)
):
    user = db.query(Utilisateur).filter_by(email=email).first()
    if user:
        token = secrets.token_urlsafe(32)
        user.reset_token = token
        db.commit()
        send_reset_email(user.email, token)

    return templates.TemplateResponse("login.html", {
        "request": request,
        "info": "Si l'email existe, un lien a été envoyé ✅"
    })

@app.get("/reset-password/{token}", response_class=HTMLResponse)
def reset_password_form(request: Request, token: str):
    return templates.TemplateResponse("reset_password.html", {"request": request, "token": token})

@app.post("/reset-password", response_class=HTMLResponse)
def reset_password(
        request: Request,
        token: str = Form(...),
        password: str = Form(...),
        db: Session = Depends(get_db)
):
    user = db.query(Utilisateur).filter_by(reset_token=token).first()
    if user:
        user.password = password
        user.reset_token = None
        db.commit()
        return templates.TemplateResponse("login.html", {
            "request": request,
            "info": "Mot de passe mis à jour avec succès ✅"
        })

    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": "Lien invalide ou expiré ❌"
    })

@app.get("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=302)
    return response

# -----------------------------
# Enregistrement du visage
# -----------------------------

@app.get("/register-face", response_class=HTMLResponse)
def get_register_face(request: Request):
    return templates.TemplateResponse("register_face.html", {"request": request})

@app.post("/register-face", response_class=HTMLResponse)
def register_face(
        request: Request,
        username: str = Form(...),
        face_image: str = Form(...),
        db: Session = Depends(get_db)
):
    try:
        user = db.query(Utilisateur).filter_by(username=username).first()
        if not user:
            return templates.TemplateResponse("login.html", {
                "request": request,
                "error": "Utilisateur non trouvé ❌"
            })

        img_np = decode_base64_image(face_image)
        if img_np is None:
            raise Exception("Image invalide")

        Image.fromarray(img_np).save("static/faces/face_2.jpg")  # Pour debug visuel

        face_encoding = extract_face_encoding(img_np)
        if face_encoding is None:
            raise Exception("Aucun visage détecté")

        user.face_encoding = json.dumps(face_encoding.tolist())
        db.commit()
        print("🧪 Shape:", img_np.shape, "Dtype:", img_np.dtype)

        return templates.TemplateResponse("login.html", {
            "request": request,
            "info": "Visage enregistré avec succès ✅"
        })

    except Exception as e:
        print("❌ Erreur enregistrement visage:", str(e))
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Erreur lors de l'enregistrement du visage ❌"
        })

# -----------------------------
# Connexion hybride
# -----------------------------
# -----------------------------
# Connexion hybride
# -----------------------------
@app.post("/verify-password", response_class=HTMLResponse)
def verify_password_step(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),

        db: Session = Depends(get_db)
):
    user = db.query(Utilisateur).filter_by(username=username).first()
    print("⚙️ USE_FACE_AUTH =", USE_FACE_AUTH)

    if not user:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Nom d'utilisateur incorrect ❌"
        })

    if user.password != password:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Mot de passe incorrect ❌"
        })

    # ✅ Authentification faciale requise
    if USE_FACE_AUTH == 1:
        return templates.TemplateResponse("verify_face.html", {
            "request": request,
            "username": user.username ,
            "user_id": user.id
        })

    # ⛔ Sinon, aller au dashboard
    return RedirectResponse(url="/dashboard", status_code=302)


@app.post("/verify-face", response_class=HTMLResponse)
def verify_face_step(
        request: Request,
        username: str = Form(...),
        face_image: str = Form(...),
        db: Session = Depends(get_db)
):
    try:
        user = db.query(Utilisateur).filter_by(username=username).first()
        if not user or not user.face_encoding:
            raise Exception("Utilisateur ou visage non trouvé")

        img_np = decode_base64_image(face_image)
        if img_np is None:
            raise Exception("Image invalide")

        current_encoding = extract_face_encoding(img_np)
        if current_encoding is None:
            raise Exception("Aucun visage détecté")

        stored_encoding = json.loads(user.face_encoding)
        match = compare_faces(current_encoding, stored_encoding)

        if not match:
            raise Exception("Visage non reconnu ❌")

        return RedirectResponse(url="/dashboard", status_code=302)

    except Exception as e:
        print("❌ ERREUR FACE:", e)
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": f"Erreur faciale: {str(e)} ❌"
        })

# -----------------------------
# Route de test faciale
# -----------------------------

@app.get("/test-face-form", response_class=HTMLResponse)
def test_face_form(request: Request):
    return templates.TemplateResponse("test_face_match.html", {"request": request})

@app.post("/test-face-match")
async def test_face_match(
    img1: UploadFile = File(...),
    img2: UploadFile = File(...)
):
    try:
        img1_bytes = await img1.read()
        img2_bytes = await img2.read()

        img1_np = np.array(Image.open(BytesIO(img1_bytes)))
        img2_np = np.array(Image.open(BytesIO(img2_bytes)))

        encoding1 = extract_face_encoding(img1_np)
        encoding2 = extract_face_encoding(img2_np)

        if encoding1 is None or encoding2 is None:
            return JSONResponse({
                "match": False,
                "distance": None,
                "message": "❌ Aucun visage détecté dans l'une des deux images."
            })

        match = compare_faces(encoding1, encoding2)
        distance = face_recognition.face_distance([encoding1], encoding2)[0]

        return JSONResponse({
            "match": match,
            "distance": float(distance),
            "message": "✅ Visages identiques" if match else "❌ Visages différents"
        })

    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/applications")
def get_all_applications(db: Session = Depends(get_db)):
    apps = db.query(Application).all()
    return apps


@app.get("/application/{id}")
def get_application(id: int, db: Session = Depends(get_db)):
    app = db.query(Application).filter(Application.id == id).first()
    if app:
        return {
            "id": app.id,
            "nom": app.nom,
            "adresse_ip": app.adresse_ip,
            "ip_base_donnees": app.ip_base_donnees,
            "mode_execution": app.mode_execution,
            "commande": app.commande,
            "active": app.active
        }
    return JSONResponse(status_code=404, content={"error": "Application non trouvée"})





class ApplicationUpdate(BaseModel):
    nom: str
    adresse_ip: str
    ip_base_donnees: str
    mode_execution: str
    commande: str | None = None
    active: bool

@app.put("/application/{id}")
def update_application(id: int, updated_app: ApplicationUpdate, db: Session = Depends(get_db)):
    app = db.query(Application).filter(Application.id == id).first()
    if not app:
        return JSONResponse(status_code=404, content={"error": "Application non trouvée"})

    app.nom = updated_app.nom
    app.adresse_ip = updated_app.adresse_ip
    app.ip_base_donnees = updated_app.ip_base_donnees
    app.mode_execution = updated_app.mode_execution
    app.commande = updated_app.commande
    app.active = updated_app.active

    db.commit()
    db.refresh(app)
    return {
        "message": "✅ Application mise à jour avec succès",
        "application": {
            "id": app.id,
            "nom": app.nom
        }
    }


@app.get("/register", response_class=HTMLResponse)
def show_register_form(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})




@app.post("/register", response_class=HTMLResponse)
def register_user(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    face_image: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    # Vérification de l'unicité
    existing_user = db.query(Utilisateur).filter(
        (Utilisateur.username == username) | (Utilisateur.email == email)
    ).first()
    if existing_user:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Nom d'utilisateur ou email déjà utilisé ❌"
        })

    # Vérification complexité du mot de passe
    if len(password) < 8 or not re.search(r"[A-Z]", password) or \
       not re.search(r"[0-9]", password) or not re.search(r"[^A-Za-z0-9]", password):
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Le mot de passe doit contenir au moins 8 caractères, une majuscule, un chiffre et un caractère spécial ❌"
        })

    # Création de l'utilisateur
    new_user = Utilisateur(username=username, email=email, password=password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)  # pour récupérer new_user.id

    # Traitement du visage si fourni
    if face_image:
        try:
            base64_data = face_image.split(",")[1]
            img_bytes = base64.b64decode(base64_data)

            # 📸 Sauvegarder l'image dans /static/faces/
            faces_dir = "static/faces"
            os.makedirs(faces_dir, exist_ok=True)
            face_image_path = os.path.join(faces_dir, f"face_{new_user.id}.jpg")
            with open(face_image_path, "wb") as f:
                f.write(img_bytes)

            # 📁 Sauvegarde temporaire pour encodage
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                tmp_file.write(img_bytes)
                tmp_file_path = tmp_file.name

            # 🔍 Encodage avec OpenCV
            detect_face_and_save_vector(tmp_file_path)

            if os.path.exists("face_encoding_opencv.json"):
                with open("face_encoding_opencv.json", "r") as f:
                    data = json.load(f)
                    encoding_str = json.dumps(data["encoding"])
                    new_user.face_encoding = encoding_str
                    db.commit()
            else:
                print("❌ Fichier face_encoding_opencv.json non généré.")

            os.remove(tmp_file_path)

        except Exception as e:
            print("❌ Erreur encodage visage :", e)

    return templates.TemplateResponse("login.html", {
        "request": request,
        "info": "Compte créé avec succès ✅ Connectez-vous"
    })
