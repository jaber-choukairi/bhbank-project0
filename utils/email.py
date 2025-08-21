import smtplib
from email.message import EmailMessage

# Remplace avec ton adresse réelle
EMAIL_SENDER = "jaberchoukairi7@gmail.com"
EMAIL_PASSWORD = "wsluwruykohwomwd"  # Pas le vrai mot de passe Gmail !

def send_reset_email(to_email: str, token: str):
    reset_link = f"http://localhost:8000/reset-password/{token}"
    subject = "Réinitialisation de votre mot de passe"
    body = f"Bonjour,\n\nVoici le lien pour réinitialiser votre mot de passe :\n{reset_link}\n\nSi vous n'avez pas demandé cela, ignorez cet email."

    msg = EmailMessage()
    msg["From"] = EMAIL_SENDER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("Email envoyé à", to_email)
    except Exception as e:
        print("Erreur d’envoi :", e)
