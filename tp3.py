import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def binariser_image(chemin_image):
    """
    Fonction complète pour binariser une image avec différentes méthodes
    """
    
    # 1. Vérifier si le fichier existe
    if not os.path.exists(chemin_image):
        print(f"Erreur: Le fichier {chemin_image} n'existe pas!")
        return
    
    # 2. Charger l'image en niveaux de gris
    print("Chargement de l'image...")
    img = cv2.imread(chemin_image, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Erreur: Impossible de charger l'image!")
        return
    
    print(f"Image chargée avec succès - Dimensions: {img.shape}")
    
    # 3. Afficher l'histogramme pour analyser la distribution des pixels
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Image Originale')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.title('Histogramme des niveaux de gris')
    plt.xlabel('Intensité')
    plt.ylabel('Nombre de pixels')
    plt.show()
    
    # 4. Différentes méthodes de binarisation
    print("\nApplication des différentes méthodes de binarisation...")
    
    # Seuillage fixe
    seuil_fixe = 127
    _, binary_fixed = cv2.threshold(img, seuil_fixe, 255, cv2.THRESH_BINARY)
    
    # Seuillage d'Otsu (automatique)
    seuil_otsu, binary_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Seuil calculé par Otsu: {seuil_otsu:.0f}")
    
    # Seuillage adaptatif (Mean)
    binary_adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
    
    # Seuillage adaptatif (Gaussian)
    binary_adaptive_gauss = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
    
    # Seuillage inverse (arrière-plan blanc, objet noir)
    _, binary_inverse = cv2.threshold(img, seuil_otsu, 255, cv2.THRESH_BINARY_INV)
    
    # 5. Affichage comparatif de toutes les méthodes
    plt.figure(figsize=(18, 12))
    
    # Image originale
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Image Originale', fontsize=12)
    plt.axis('off')
    
    # Seuil fixe
    plt.subplot(2, 3, 2)
    plt.imshow(binary_fixed, cmap='gray')
    plt.title(f'Seuil Fixe (seuil = {seuil_fixe})', fontsize=12)
    plt.axis('off')
    
    # Otsu
    plt.subplot(2, 3, 3)
    plt.imshow(binary_otsu, cmap='gray')
    plt.title(f'Seuillage d\'Otsu (seuil = {seuil_otsu:.0f})', fontsize=12)
    plt.axis('off')
    
    # Adaptatif Mean
    plt.subplot(2, 3, 4)
    plt.imshow(binary_adaptive_mean, cmap='gray')
    plt.title('Adaptatif (Mean)', fontsize=12)
    plt.axis('off')
    
    # Adaptatif Gaussian
    plt.subplot(2, 3, 5)
    plt.imshow(binary_adaptive_gauss, cmap='gray')
    plt.title('Adaptatif (Gaussian)', fontsize=12)
    plt.axis('off')
    
    # Inverse
    plt.subplot(2, 3, 6)
    plt.imshow(binary_inverse, cmap='gray')
    plt.title('Seuillage Inverse (Otsu)', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 6. Sauvegarder les résultats
    dossier_sortie = os.path.dirname(chemin_image)
    nom_fichier = os.path.splitext(os.path.basename(chemin_image))[0]
    
    # Créer les noms de fichiers de sortie
    fichiers_sortie = {
        'seuil_fixe': f"{dossier_sortie}\\{nom_fichier}_binaire_seuil_fixe.jpg",
        'otsu': f"{dossier_sortie}\\{nom_fichier}_binaire_otsu.jpg",
        'adaptatif_mean': f"{dossier_sortie}\\{nom_fichier}_binaire_adaptatif_mean.jpg",
        'adaptatif_gauss': f"{dossier_sortie}\\{nom_fichier}_binaire_adaptatif_gauss.jpg",
        'inverse': f"{dossier_sortie}\\{nom_fichier}_binaire_inverse.jpg"
    }
    
    # Sauvegarder les images
    print(f"\nSauvegarde des images binarisées...")
    cv2.imwrite(fichiers_sortie['seuil_fixe'], binary_fixed)
    cv2.imwrite(fichiers_sortie['otsu'], binary_otsu)
    cv2.imwrite(fichiers_sortie['adaptatif_mean'], binary_adaptive_mean)
    cv2.imwrite(fichiers_sortie['adaptatif_gauss'], binary_adaptive_gauss)
    cv2.imwrite(fichiers_sortie['inverse'], binary_inverse)
    
    print("Images sauvegardées:")
    for nom, fichier in fichiers_sortie.items():
        print(f"  - {nom}: {fichier}")
    
    # 7. Retourner les images pour utilisation ultérieure
    return {
        'originale': img,
        'seuil_fixe': binary_fixed,
        'otsu': binary_otsu,
        'adaptatif_mean': binary_adaptive_mean,
        'adaptatif_gauss': binary_adaptive_gauss,
        'inverse': binary_inverse
    }

def analyser_resultats(images_dict):
    """
    Fonction pour analyser les résultats de binarisation
    """
    if images_dict is None:
        return
    
    print("\n=== ANALYSE DES RÉSULTATS ===")
    
    for nom, img in images_dict.items():
        if nom == 'originale':
            continue
            
        # Compter les pixels blancs et noirs
        pixels_blancs = np.sum(img == 255)
        pixels_noirs = np.sum(img == 0)
        total_pixels = img.size
        
        pourcentage_blancs = (pixels_blancs / total_pixels) * 100
        pourcentage_noirs = (pixels_noirs / total_pixels) * 100
        
        print(f"\n{nom.upper()}:")
        print(f"  Pixels blancs: {pixels_blancs} ({pourcentage_blancs:.1f}%)")
        print(f"  Pixels noirs: {pixels_noirs} ({pourcentage_noirs:.1f}%)")

def main():
    """
    Fonction principale
    """
    # Chemin vers votre image - modifié pour correspondre au nouveau chemin
    chemin_image = "C:\\Users\\USER-PC\\jupyter\\profile.jpg"
    
    print("=== BINARISATION D'IMAGE AVEC OPENCV ===")
    print(f"Image à traiter: {chemin_image}")
    
    # Binariser l'image
    images_resultat = binariser_image(chemin_image)
    
    # Analyser les résultats
    analyser_resultats(images_resultat)
    
    # Option pour tester différents seuils
    print("\n=== TEST DE DIFFÉRENTS SEUILS FIXES ===")
    
    if images_resultat is not None:
        img_originale = images_resultat['originale']
        
        seuils_test = [50, 100, 127, 150, 200]
        
        plt.figure(figsize=(15, 6))
        plt.subplot(2, 3, 1)
        plt.imshow(img_originale, cmap='gray')
        plt.title('Image Originale')
        plt.axis('off')
        
        for i, seuil in enumerate(seuils_test):
            _, img_seuil = cv2.threshold(img_originale, seuil, 255, cv2.THRESH_BINARY)
            plt.subplot(2, 3, i+2)
            plt.imshow(img_seuil, cmap='gray')
            plt.title(f'Seuil = {seuil}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print("\n=== TRAITEMENT TERMINÉ ===")

# Version simplifiée pour test rapide
def binarisation_simple(chemin_image=None):
    """
    Version simplifiée pour tester rapidement
    """
    if chemin_image is None:
        chemin_image = "C:\Users\USER-PC\jupyter\\vick.jpg"
    
    # Charger l'image
    img = cv2.imread(chemin_image, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Erreur: Impossible de charger {chemin_image}")
        return
    
    # Binarisation Otsu (méthode recommandée)
    _, img_binaire = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Affichage simple
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Image Originale')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_binaire, cmap='gray')
    plt.title('Image Binarisée (Otsu)')
    plt.axis('off')
    
    plt.show()
    
    return img_binaire

# Exécution du programme principal
if __name__ == "__main__":
    main()
    
    # Décommentez cette ligne pour un test rapide :
    # binarisation_simple()