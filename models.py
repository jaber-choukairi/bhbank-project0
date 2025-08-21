from sqlalchemy import Column, Integer, String, Boolean, Text
from database import Base

class Utilisateur(Base):
    __tablename__ = "utilisateurs"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password = Column(String(100), nullable=False)
    reset_token = Column(String(255), nullable=True)  # Pour la récupération de mot de passe
    face_encoding = Column(Text, nullable=True)       # Encodage facial stocké au format texte




class Application(Base):
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String(100))
    adresse_ip = Column(String(45))
    ip_base_donnees = Column(String(45))
    mode_execution = Column(String(50))
    commande = Column(Text)
    active = Column(Boolean, default=False)
