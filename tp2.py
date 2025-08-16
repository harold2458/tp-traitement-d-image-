import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

def binariser_image_unique(img, nom_image):
    """
    Fonction pour binariser une seule image avec différentes méthodes
    """
    if img is None:
        print(f"Erreur: Impossible de traiter l'image {nom_image}")
        return None
    
    print(f"Traitement de: {nom_image} - Dimensions: {img.shape}")
    
    # Différentes méthodes de binarisation
    seuil_fixe = 127
    _, binary_fixed = cv2.threshold(img, seuil_fixe, 255, cv2.THRESH_BINARY)
    
    # Seuillage d'Otsu (automatique)
    seuil_otsu, binary_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Seuillage adaptatif (Mean)
    binary_adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
    
    # Seuillage adaptatif (Gaussian)
    binary_adaptive_gauss = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
    
    # Seuillage inverse
    _, binary_inverse = cv2.threshold(img, seuil_otsu, 255, cv2.THRESH_BINARY_INV)
    
    return {
        'originale': img,
        'seuil_fixe': binary_fixed,
        'otsu': binary_otsu,
        'adaptatif_mean': binary_adaptive_mean,
        'adaptatif_gauss': binary_adaptive_gauss,
        'inverse': binary_inverse,
        'seuil_otsu_valeur': seuil_otsu
    }

def traiter_dossier_images(dossier_entree, dossier_sortie=None, extensions=['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']):
    """
    Traiter toutes les images d'un dossier
    """
    # Créer le dossier de sortie s'il n'existe pas
    if dossier_sortie is None:
        dossier_sortie = os.path.join(dossier_entree, "images_binarisees")
    
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    
    # Trouver toutes les images
    fichiers_images = []
    for ext in extensions:
        fichiers_images.extend(glob.glob(os.path.join(dossier_entree, ext)))
        fichiers_images.extend(glob.glob(os.path.join(dossier_entree, ext.upper())))
    
    if not fichiers_images:
        print(f"Aucune image trouvée dans le dossier: {dossier_entree}")
        return []
    
    print(f"Trouvé {len(fichiers_images)} image(s) à traiter:")
    for fichier in fichiers_images:
        print(f"  - {os.path.basename(fichier)}")
    
    resultats_globaux = []
    
    for i, chemin_image in enumerate(fichiers_images, 1):
        print(f"\n=== IMAGE {i}/{len(fichiers_images)} ===")
        
        # Charger l'image
        img = cv2.imread(chemin_image, cv2.IMREAD_GRAYSCALE)
        nom_image = os.path.basename(chemin_image)
        nom_sans_extension = os.path.splitext(nom_image)[0]
        
        # Binariser l'image
        resultats = binariser_image_unique(img, nom_image)
        
        if resultats is None:
            continue
        
        # Sauvegarder les résultats
        sauvegarder_resultats(resultats, nom_sans_extension, dossier_sortie)
        
        # Ajouter aux résultats globaux
        resultats['nom_fichier'] = nom_image
        resultats['chemin_original'] = chemin_image
        resultats_globaux.append(resultats)
    
    return resultats_globaux

def traiter_liste_images(liste_chemins, dossier_sortie):
    """
    Traiter une liste spécifique d'images
    """
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    
    resultats_globaux = []
    
    for i, chemin_image in enumerate(liste_chemins, 1):
        print(f"\n=== IMAGE {i}/{len(liste_chemins)} ===")
        
        if not os.path.exists(chemin_image):
            print(f"Fichier non trouvé: {chemin_image}")
            continue
        
        # Charger l'image
        img = cv2.imread(chemin_image, cv2.IMREAD_GRAYSCALE)
        nom_image = os.path.basename(chemin_image)
        nom_sans_extension = os.path.splitext(nom_image)[0]
        
        # Binariser l'image
        resultats = binariser_image_unique(img, nom_image)
        
        if resultats is None:
            continue
        
        # Sauvegarder les résultats
        sauvegarder_resultats(resultats, nom_sans_extension, dossier_sortie)
        
        # Ajouter aux résultats globaux
        resultats['nom_fichier'] = nom_image
        resultats['chemin_original'] = chemin_image
        resultats_globaux.append(resultats)
    
    return resultats_globaux

def sauvegarder_resultats(resultats, nom_sans_extension, dossier_sortie):
    """
    Sauvegarder toutes les versions binarisées d'une image
    """
    methodes = ['seuil_fixe', 'otsu', 'adaptatif_mean', 'adaptatif_gauss', 'inverse']
    
    for methode in methodes:
        nom_fichier = f"{nom_sans_extension}_{methode}.jpg"
        chemin_sortie = os.path.join(dossier_sortie, nom_fichier)
        cv2.imwrite(chemin_sortie, resultats[methode])
    
    print(f"  Images sauvegardées dans: {dossier_sortie}")

def afficher_comparaison_multiple(resultats_globaux, max_images=6):
    """
    Afficher une comparaison de plusieurs images traitées
    """
    if not resultats_globaux:
        print("Aucun résultat à afficher")
        return
    
    nombre_images = min(len(resultats_globaux), max_images)
    
    # Affichage des images originales
    plt.figure(figsize=(20, 4*nombre_images))
    
    for i in range(nombre_images):
        resultats = resultats_globaux[i]
        
        # Image originale
        plt.subplot(nombre_images, 6, i*6 + 1)
        plt.imshow(resultats['originale'], cmap='gray')
        plt.title(f"{resultats['nom_fichier']}\nOriginale")
        plt.axis('off')
        
        # Seuil fixe
        plt.subplot(nombre_images, 6, i*6 + 2)
        plt.imshow(resultats['seuil_fixe'], cmap='gray')
        plt.title("Seuil Fixe")
        plt.axis('off')
        
        # Otsu
        plt.subplot(nombre_images, 6, i*6 + 3)
        plt.imshow(resultats['otsu'], cmap='gray')
        plt.title(f"Otsu ({resultats['seuil_otsu_valeur']:.0f})")
        plt.axis('off')
        
        # Adaptatif Mean
        plt.subplot(nombre_images, 6, i*6 + 4)
        plt.imshow(resultats['adaptatif_mean'], cmap='gray')
        plt.title("Adaptatif Mean")
        plt.axis('off')
        
        # Adaptatif Gaussian
        plt.subplot(nombre_images, 6, i*6 + 5)
        plt.imshow(resultats['adaptatif_gauss'], cmap='gray')
        plt.title("Adaptatif Gauss")
        plt.axis('off')
        
        # Inverse
        plt.subplot(nombre_images, 6, i*6 + 6)
        plt.imshow(resultats['inverse'], cmap='gray')
        plt.title("Inverse")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def generer_rapport(resultats_globaux, dossier_sortie):
    """
    Générer un rapport détaillé des traitements
    """
    if not resultats_globaux:
        return
    
    rapport_path = os.path.join(dossier_sortie, "rapport_binarisation.txt")
    
    with open(rapport_path, 'w', encoding='utf-8') as f:
        f.write("=== RAPPORT DE BINARISATION D'IMAGES ===\n\n")
        f.write(f"Nombre d'images traitées: {len(resultats_globaux)}\n")
        f.write(f"Date de traitement: {str(np.datetime64('now'))}\n\n")
        
        for i, resultats in enumerate(resultats_globaux, 1):
            f.write(f"\n--- IMAGE {i}: {resultats['nom_fichier']} ---\n")
            f.write(f"Chemin original: {resultats['chemin_original']}\n")
            f.write(f"Dimensions: {resultats['originale'].shape}\n")
            f.write(f"Seuil Otsu calculé: {resultats['seuil_otsu_valeur']:.0f}\n")
            
            # Statistiques pour chaque méthode
            for methode in ['seuil_fixe', 'otsu', 'adaptatif_mean', 'adaptatif_gauss', 'inverse']:
                img = resultats[methode]
                pixels_blancs = np.sum(img == 255)
                pixels_noirs = np.sum(img == 0)
                total_pixels = img.size
                pourcentage_blancs = (pixels_blancs / total_pixels) * 100
                
                f.write(f"\n  {methode.upper()}:\n")
                f.write(f"    Pixels blancs: {pixels_blancs} ({pourcentage_blancs:.1f}%)\n")
                f.write(f"    Pixels noirs: {pixels_noirs} ({100-pourcentage_blancs:.1f}%)\n")
    
    print(f"\nRapport généré: {rapport_path}")

def main():
    """
    Fonction principale avec menu interactif
    """
    print("=== BINARISATION MULTIPLE D'IMAGES AVEC OPENCV ===\n")
    
    print("Choisissez une option:")
    print("1. Traiter toutes les images d'un dossier")
    print("2. Traiter une liste spécifique d'images")
    print("3. Exemple avec images de test")
    
    choix = input("\nVotre choix (1/2/3): ").strip()
    
    if choix == "1":
        # Option 1: Traiter un dossier complet
        dossier_entree = input("Chemin du dossier contenant les images: ").strip()
        if not dossier_entree:
            dossier_entree = "C:\\Users\\USER-PC\\TP traitement d'image"
        
        dossier_sortie = input("Dossier de sortie (laisser vide pour créer un sous-dossier): ").strip()
        
        print(f"\nTraitement du dossier: {dossier_entree}")
        resultats = traiter_dossier_images(dossier_entree, dossier_sortie if dossier_sortie else None)
        
    elif choix == "2":
        # Option 2: Liste spécifique d'images
        print("\nEntrez les chemins des images (un par ligne, ligne vide pour terminer):")
        liste_images = []
        while True:
            chemin = input("Chemin image: ").strip()
            if not chemin:
                break
            liste_images.append(chemin)
        
        if not liste_images:
            print("Aucune image spécifiée!")
            return
        
        dossier_sortie = input("Dossier de sortie: ").strip()
        if not dossier_sortie:
            dossier_sortie = "images_binarisees"
        
        resultats = traiter_liste_images(liste_images, dossier_sortie)
        
    elif choix == "3":
        # Option 3: Exemple avec images de test
        liste_exemple = [
            "C:\\Users\\USER-PC\\TP traitement d'image\\profile.jpg",
            "C:\\Users\\USER-PC\\TP traitement d'image\\image1.jpg",
            "C:\\Users\\USER-PC\\TP traitement d'image\\image2.png"
        ]
        
        print(f"\nExemple avec {len(liste_exemple)} images:")
        for img in liste_exemple:
            print(f"  - {img}")
        
        dossier_sortie = "C:\\Users\\USER-PC\\TP traitement d'image\\resultats_binarisation"
        resultats = traiter_liste_images(liste_exemple, dossier_sortie)
    
    else:
        print("Choix invalide!")
        return
    
    # Affichage des résultats
    if resultats:
        print(f"\n=== TRAITEMENT TERMINÉ ===")
        print(f"Nombre d'images traitées avec succès: {len(resultats)}")
        
        # Générer le rapport
        dossier_rapport = os.path.dirname(resultats[0]['chemin_original']) if resultats else "."
        if choix == "1" and 'dossier_sortie' in locals():
            dossier_rapport = dossier_sortie or os.path.join(dossier_entree, "images_binarisees")
        elif choix in ["2", "3"]:
            dossier_rapport = dossier_sortie
        
        generer_rapport(resultats, dossier_rapport)
        
        # Demander si afficher la comparaison
        afficher = input("\nVoulez-vous afficher la comparaison visuelle? (o/n): ").strip().lower()
        if afficher in ['o', 'oui', 'y', 'yes']:
            max_img = int(input(f"Combien d'images afficher au maximum? (1-{len(resultats)}): ") or "3")
            afficher_comparaison_multiple(resultats, max_img)
    
    else:
        print("\nAucune image n'a pu être traitée!")

# Fonction utilitaire pour traitement rapide
def traitement_rapide(dossier_ou_liste, dossier_sortie="images_binarisees"):
    """
    Fonction pour traitement rapide sans menu interactif
    
    Paramètres:
    - dossier_ou_liste: soit un chemin de dossier, soit une liste de fichiers
    - dossier_sortie: dossier où sauvegarder les résultats
    """
    if isinstance(dossier_ou_liste, str):
        # C'est un dossier
        resultats = traiter_dossier_images(dossier_ou_liste, dossier_sortie)
    else:
        # C'est une liste de fichiers
        resultats = traiter_liste_images(dossier_ou_liste, dossier_sortie)
    
    if resultats:
        generer_rapport(resultats, dossier_sortie)
        afficher_comparaison_multiple(resultats, min(3, len(resultats)))
    
    return resultats

# Exécution du programme
if __name__ == "__main__":
    main()
    
    # Exemple d'utilisation de la fonction rapide (décommenter pour utiliser):
    # resultats = traitement_rapide("C:\\Users\\USER-PC\\TP traitement d'image")
    
    # Ou avec une liste spécifique:
    # mes_images = [
    #     "C:\\Users\\USER-PC\\TP traitement d'image\\profile.jpg",
    #     "C:\\Users\\USER-PC\\TP traitement d'image\\image2.jpg"
    # ]
    # resultats = traitement_rapide(mes_images, "mes_resultats")