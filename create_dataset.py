import os
import numpy as np
import face_recognition
from google.colab import files

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

print("--- REGISTRATSIYA (EMBEDDING SAQLASH) ---")

# 1. Rasm yuklash
print("Iltimos, inson rasmini yuklang:")
uploaded = files.upload()

if uploaded:
    img_name = list(uploaded.keys())[0]
    
    # Rasmni yuklash va embedding olish
    image = face_recognition.load_image_file(img_name)
    encodings = face_recognition.face_encodings(image)
    
    if len(encodings) > 0:
        # 2. Ism kiritish
        person_name = input("Bu rasmdagi insonning ismi (masalan: Ali): ").strip()
        
        if person_name:
            # Mavjud .npy fayllarni tekshirish
            existing_files = os.listdir(DATASET_DIR)
            count = 1
            
            while f"{person_name}_{count}.npy" in existing_files:
                count += 1
                
            new_filename = f"{person_name}_{count}.npy"
            final_path = os.path.join(DATASET_DIR, new_filename)
            
            # Embeddingni .npy formatida saqlash (Rasm emas, massiv saqlanadi)
            np.save(final_path, encodings[0])
            print(f"\n[MUVAFFAQIYATLI] Yuz kodi muvaffaqiyatli saqlandi: {final_path}")
        else:
            print("[XATO] Ism kiritilmadi.")
    else:
        print("[XATO] Yuklangan rasmdan inson yuzi topilmadi!")
    
    # Vaqtinchalik rasmni o'chirish
    os.remove(img_name)
