import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Chargement EasyOCR ---
try:
    import easyocr
    EASY_AVAILABLE = True
except Exception as e:
    EASY_AVAILABLE = False
    print("EasyOCR non disponible. Installez-le avec : pip install easyocr")
    print("Erreur :", e)

# --- Fonctions utilitaires ---
def imshow(title, image, size=(8,6)):
    plt.figure(figsize=size)
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def load_gray(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image introuvable : {path}")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Impossible de charger l'image (format non supporté ?)")
    return img

def denoise(img, method='bilateral'):
    if method == 'gaussian':
        return cv2.GaussianBlur(img, (3,3), 0)
    return cv2.bilateralFilter(img, 7, 50, 50)

def binarize(img):
    eq = cv2.equalizeHist(img)
    adapt = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 35, 15)
    _, otsu = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    fused = cv2.bitwise_and(adapt, otsu)
    return fused, {'adapt': adapt, 'otsu': otsu}

def deskew(binary):
    inv = binary if np.mean(binary) > 127 else cv2.bitwise_not(binary)
    coords = np.column_stack(np.where(inv > 0))
    angle = 0.0
    if len(coords) > 0:
        rect = cv2.minAreaRect(coords.astype(np.float32))
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
    (h, w) = binary.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle

# --- Pipeline de prétraitement ---
def preprocess(path):
    gray = load_gray(path)
    imshow('Original (gris)', gray)
    d = denoise(gray, method='bilateral')
    imshow('Dénoyage (bilateral)', d)
    b, parts = binarize(d)
    imshow('Binarisation fusionnée', b)
    imshow('Seuillage adaptatif', parts['adapt'])
    imshow('Otsu', parts['otsu'])
    desk, angle = deskew(b)
    print(f'Angle estimé : {angle:.2f}°')
    imshow('Après deskew', desk)
    return gray, d, b, desk, angle

# --- Segmentation des lignes ---
def segment_lines(binary, min_line_height=8, pad=6):
    img = binary.copy()
    if np.mean(img) < 127:
        img = cv2.bitwise_not(img)
    
    proj = np.sum(255 - img, axis=1)
    proj_norm = (proj - proj.min()) / (np.ptp(proj) + 1e-6)

    # Plus permissif
    thresh = np.mean(proj_norm) * 0.3
    lines = []
    in_line = False
    start = 0
    
    for i, v in enumerate(proj_norm):
        if v > thresh and not in_line:
            in_line = True
            start = i
        elif v <= thresh and in_line:
            end = i
            if end - start >= min_line_height:
                s = max(0, start - pad)
                e = min(img.shape[0], end + pad)
                lines.append((s, e))
            in_line = False
    
    if in_line:
        end = len(proj_norm)-1
        if end - start >= min_line_height:
            s = max(0, start - pad)
            e = min(img.shape[0], end + pad)
            lines.append((s, e))
    
    crops = []
    for (s,e) in lines:
        crop = binary[s:e, :]
        crops.append(((s,e), crop))
    
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for (s,e) in lines:
        cv2.rectangle(vis, (0,s), (vis.shape[1]-1, e), (0,0,255), 1)
    imshow('Segmentation des lignes', vis, size=(10,8))
    
    return crops, proj_norm, lines

# --- OCR ---
def build_reader(lang_list=['fr','en']):
    if not EASY_AVAILABLE:
        raise ImportError("EasyOCR non disponible. Installez-le.")
    return easyocr.Reader(lang_list, gpu=False)

def ocr_lines(reader, line_crops):
    texts = []
    for idx, (_, crop) in enumerate(line_crops, 1):
        if len(crop.shape) == 2:
            bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        else:
            bgr = crop
        result = reader.readtext(bgr, detail=0, paragraph=True, contrast_ths=0.5, adjust_contrast=0.7)
        line_text = ' '.join(result).strip()
        texts.append(line_text)
        print(f"Ligne {idx} : {line_text}")
    return texts

# --- Pipeline complet ---
def run_pipeline(image_path, langs=['fr','en']):
    gray, den, bin_img, desk, angle = preprocess(image_path)

    if EASY_AVAILABLE:
        reader = build_reader(langs)

        # 🔹 Test direct sur toute l'image
        result_full = reader.readtext(cv2.cvtColor(desk, cv2.COLOR_GRAY2BGR),
                                      detail=0, paragraph=True,
                                      contrast_ths=0.5, adjust_contrast=0.7)
        transcript_full = ' '.join(result_full)
        print("\n=== OCR direct (sans segmentation) ===\n", transcript_full)

        # 🔹 Segmentation améliorée
        line_crops, proj, boxes = segment_lines(desk)
        if len(line_crops) > 0:
            lines_text = ocr_lines(reader, line_crops)
            transcript_seg = '\n'.join(lines_text)
            print("\n=== OCR après segmentation ===\n", transcript_seg)
        else:
            transcript_seg = ''
            print("⚠️ Aucune ligne détectée.")

        return transcript_full, transcript_seg
    else:
        print("EasyOCR indisponible.")
        return '', ''

# --- Test ---
IMAGE_PATH = r"C:\Users\USER-PC\TP traitement d'image\profile.jpg"

if os.path.exists(IMAGE_PATH):
    transcript_full, transcript_seg = run_pipeline(IMAGE_PATH, langs=['fr','en'])
else:
    print("⚠️ Chemin d'image introuvable. Modifiez IMAGE_PATH.")

# --- Métrique CER ---
def levenshtein(a, b):
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0]*n
        for j in range(1, n + 1):
            add, delete, change = previous[j] + 1, current[j-1] + 1, previous[j-1]
            if a[j-1] != b[i-1]:
                change += 1
            current[j] = min(add, delete, change)
    return current[n]
