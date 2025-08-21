face_encoding = extract_face_encoding(img_np)

if face_encoding is None:
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": "Aucun visage détecté ❌"
    })

# Sauvegarde JSON de 128 valeurs
user.face_encoding = str(face_encoding)
db.commit()
