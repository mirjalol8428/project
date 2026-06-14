import os
import cv2
import numpy as np
import face_recognition
import matplotlib.pyplot as plt
from google.colab import files

DATASET_DIR = "dataset"

known_encodings = []
known_names = []

# 1. .npy formatidagi tayyor yuz kodlarini yuklash (Juda tez yuklanadi)
if not os.path.exists(DATASET_DIR) or len(os.listdir(DATASET_DIR)) == 0:
    print("[XATO] Dataset bo'sh! Avval 1-kod orqali yuzlarni ro'yxatga oling.")
else:
    print("[INFO] .npy bazasi xotiraga yuklanmoqda...")
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith('.npy'):
            file_path = os.path.join(DATASET_DIR, filename)
            
            # Fayl nomidan ismni ajratish ("Ali_1.npy" -> "Ali")
            name_part = filename.split('_')[0]
            
            # .npy faylni o'qish
            embedding = np.load(file_path)
            
            known_encodings.append(embedding)
            known_names.append(name_part)
            
    print(f"[INFO] Baza tayyor! Jami {len(known_names)} ta yuz kodi yuklandi.\n")

    # 2. Tekshirish uchun yangi rasm yuklash
    print("--- INSONNI TEZKOR ANIQLASH (TEST) ---")
    uploaded_test = files.upload()

    if uploaded_test:
        test_img_name = list(uploaded_test.keys())[0]
        
        test_image = face_recognition.load_image_file(test_img_name)
        face_locations = face_recognition.face_locations(test_image)
        face_encodings = face_recognition.face_encodings(test_image, face_locations)
        
        display_img = cv2.imread(test_img_name)
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Tayyor embeddinglar bilan bir zumda solishtirish
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            name = "Noma'lum"
            
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
            
            # Natijani chizish
            cv2.rectangle(display_img, (left, top), (right, bottom), (0, 255, 0), 4)
            cv2.putText(display_img, name, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            print(f"[NATIJA] Aniqlangan inson: {name}")

        # 3. Ko'rsatish
        plt.figure(figsize=(10, 8))
        plt.imshow(display_img)
        plt.axis('off')
        plt.show()
        
        os.remove(test_img_name)
