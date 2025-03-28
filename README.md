# M
text extraction
import cv2
import pytesseract
from matplotlib import pyplot as plt

# Charger l'image (chemin mis à jour pour le dossier Téléchargements)
image_path = "/home/votre_nom_utilisateur/Téléchargements/LE PETIT PRINCE.jpeg"
image = cv2.imread(image_path)

# Vérifier si l'image a été chargée correctement
if image is None:
    raise FileNotFoundError(f"L'image n'a pas été trouvée au chemin : {image_path}")

# Convertir en niveaux de gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Appliquer un filtre pour améliorer la lisibilité
gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Utiliser Tesseract pour détecter le texte
custom_config = r'--oem 3 --psm 6'
extracted_text = pytesseract.image_to_string(gray, config=custom_config, lang='fra')

# Afficher l'image et le texte extrait
plt.figure(figsize=(10, 5))
plt.imshow(gray, cmap='gray')
plt.title("Image prétraitée")
plt.axis("off")
plt.show()

print("Texte détecté : \n")
print(extracted_text)
