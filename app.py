import os
import cv2
import numpy as np
import face_recognition
import streamlit as st
from PIL import Image

# Dataset saqlanadigan papka
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# 1. VEB-SAYTNING UMUMIY FRONTEND STRUKTURASI VA DIZAYNI (CSS)
st.set_page_config(page_title="Premium Yuz Tanish Tizimi", layout="centered")

# Maxsus UI/UX dizayn (CSS o'rnatish)
st.markdown("""
    <style>
    /* Umumiy fon va matn ranglari */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Sarlavha dizayni (Glassmorphism effekti bilan) */
    .main-title {
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(45deg, #1e90ff, #00ff7f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        color: #8892b0;
        margin-bottom: 30px;
    }
    
    /* Ma'lumotlar kartochkasi dizayni */
    .custom-card {
        background-color: #1b2430;
        border-radius: 12px;
        padding: 25px;
        border-left: 5px solid #1e90ff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 25px;
    }
    
    /* Yon panel (Sidebar) dizayni */
    section[data-testid="stSidebar"] {
        background-color: #111625 !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Tugmalarni chiroyli qilish */
    div.stButton > button {
        background: linear-gradient(135deg, #1e90ff 0%, #0077e6 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(30, 144, 255, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(30, 144, 255, 0.5) !important;
        background: linear-gradient(135deg, #0077e6 0%, #1e90ff 100%) !important;
    }
    
    /* O'chirish tugmasi uchun qizil rang */
    div[data-testid="stVerticalBlock"] > div:nth-child(2) div.stButton > button {
        background: linear-gradient(135deg, #ff4b4b 0%, #bf1f1f 100%) !important;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Bosh sarlavhalar (Frontend vizuali)
st.markdown("<h1 class='main-title'>👤 AI FACE RECOGNITION SYSTEM</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Sun'iy intellektga asoslangan yuzni aniqlash va dataset boshqaruv tizimi</p>", unsafe_allow_html=True)

# Yon panel navigatsiyasi (Endi 3 ta sahifa bor)
st.sidebar.markdown("<h2 style='text-align:center; color:#1e90ff;'>🧭 MENYU</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("Kerakli sahifani tanlang:", ["📥 Dataset Yozuvchi", "📦 Dataset Ro'yxati", "🔍 Main (Yuz Tanish)"])

# Footer ma'lumoti
st.sidebar.markdown("---")
st.sidebar.markdown("<p style='text-align:center; color:#8892b0; font-size:0.8rem;'>Dasturchi: Kurs ishi / Loyiha</p>", unsafe_allow_html=True)


# ==============================================================================
# 1-SAHIFA: DATASET YOZUVCHI
# ==============================================================================
if page == "📥 Dataset Yozuvchi":
    st.markdown("""
        <div class='custom-card'>
            <h3>📥 1-Sahifa: Datasetni shakllantirish</h3>
            <p style='color:#a0aec0;'>Tizim tanishi kerak bo'lgan insonlarning ismini yozing va rasmini yuklang. 
            Dastur yuz modelini (.npy) biometrik formatda saqlab boradi.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Kiritish maydonlari
    person_name = st.text_input("👤 Inson ismini kiriting (masalan: Ali, Ustoz):").strip()
    uploaded_file = st.file_uploader("📸 Rasm yuklang (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("💾 BAZAGA SAQLASH"):
        if not person_name:
            st.error("❌ Xatolik: Iltimos, inson ismini kiriting!")
        elif not uploaded_file:
            st.error("❌ Xatolik: Iltimos, rasm faylini yuklang!")
        else:
            with st.spinner("Yuz tahlil qilinmoqda, iltimos kuting..."):
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                # Biometrik yuz kodini olish
                encodings = face_recognition.face_encodings(image_np)
                
                if len(encodings) > 0:
                    existing_files = os.listdir(DATASET_DIR)
                    count = 1
                    while f"{person_name}_{count}.npy" in existing_files:
                        count += 1
                        
                    final_path = os.path.join(DATASET_DIR, f"{person_name}_{count}.npy")
                    np.save(final_path, encodings[0])
                    
                    st.balloons() # Ekrandan chiroyli sharlar uchib chiqadi
                    st.success(f"🎉 Muvaffaqiyatli! '{person_name}' tizim bazasiga qo'shildi. Fayl: {person_name}_{count}.npy")
                else:
                    st.error("❌ Xatolik: Yuklangan rasmdan yuz aniqlanmadi! Yuz aniq ko'ringan rasmdan foydalaning.")

# ==============================================================================
# 2-SAHIFA: DATASET RO'YXATI (YANGI QO'SHILGAN SAHIFA)
# ==============================================================================
elif page == "📦 Dataset Ro'yxati":
    st.markdown("""
        <div class='custom-card' style='border-left-color: #ffaa00;'>
            <h3>📦 2-Sahifa: Dataset Ma'lumotlar Ro'yxati</h3>
            <p style='color:#a0aec0;'>Hozirgi vaqtda tizim xotirasida (dataset papkasida) mavjud bo'lgan 
            barcha biometrik andozalar (.npy fayllari) ro'yxati.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Papkadagi barcha .npy fayllarni qidirish
    all_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.npy')]
    
    if len(all_files) == 0:
        st.warning("⚠️ Tizim bazasi hozircha butkul bo'sh! Iltimos, avval 'Dataset Yozuvchi' bo'limidan insonlarni qo'shing.")
    else:
        st.subheader(f"📊 Jami andozalar soni: {len(all_files)} ta")
        
        # Chiroyli jadval ko'rinishida chiqarish uchun ma'lumot yig'ish
        display_list = []
        for index, filename in enumerate(all_files, start=1):
            name_part = filename.split('_')[0]
            display_list.append({"T/r": index, "Inson Ismi": name_part, "Fayl Nomi": filename})
            
        # Jadvalni ekranga chiqarish
        st.table(display_list)
        
        st.markdown("---")
        st.subheader("🗑️ Ma'lumotni o'chirish")
        
        # O'chirish uchun faylni tanlash menyusi
        file_to_delete = st.selectbox("O'chirmoqchi bo'lgan andozani tanlang:", all_files)
        
        if st.button("❌ BAZADAN O'CHIRISH"):
            try:
                os.remove(os.path.join(DATASET_DIR, file_to_delete))
                st.success(f"🗑️ '{file_to_delete}' fayli bazadan butunlay o'chirib tashlandi!")
                st.rerun() # Sahifani yangilash (fayl ro'yxatdan yo'qolishi uchun)
            except Exception as e:
                st.error(f"Xatolik yuz berdi: {e}")

# ==============================================================================
# 3-SAHIFA: MAIN (YUZ TANISH)
# ==============================================================================
elif page == "🔍 Main (Yuz Tanish)":
    st.markdown("""
        <div class='custom-card' style='border-left-color: #00ff7f;'>
            <h3>🔍 3-Sahifa: Main - Yuzni Aniqlash Tizimi</h3>
            <p style='color:#a0aec0;'>Yangi rasm yuklang. Tizim uni .npy formatidagi biometrik ma'lumotlar 
            bilan millisekundlar ichida solishtirib, yuz atrofiga skaner to'rtburchagini chizadi.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # .npy bazasini yuklash
    known_encodings = []
    known_names = []
    
    if os.path.exists(DATASET_DIR):
        for filename in os.listdir(DATASET_DIR):
            if filename.endswith('.npy'):
                name_part = filename.split('_')[0]
                embedding = np.load(os.path.join(DATASET_DIR, filename))
                known_encodings.append(embedding)
                known_names.append(name_part)
                
    if len(known_names) == 0:
        st.warning("⚠️ Diqqat: Tizim bazasi hozircha bo'sh! Avval 'Dataset Yozuvchi' sahifasida insonlarni qo'shing.")
    else:
        # Mini status-bar
        st.markdown(f"""
            <div style='background-color:#111625; padding:10px; border-radius:8px; margin-bottom:15px; text-align:center;'>
                🟢 <span style='color:#00ff7f; font-weight:bold;'>Tizim Holati: Aktif</span> | 📦 Bazadagi jami andozalar: <b>{len(known_names)} ta</b>
            </div>
            """, unsafe_allow_html=True)
            
        test_file = st.file_uploader("🖼️ Tekshirish uchun istalgan yeni rasmni yuklang:", type=["jpg", "jpeg", "png"])
        
        if test_file:
            image = Image.open(test_file)
            image_np = np.array(image)
            display_img = image_np.copy()
            
            with st.spinner("🤖 Sun'iy intellekt yuzlarni skanerlamoqda..."):
                face_locations = face_recognition.face_locations(image_np)
                face_encodings = face_recognition.face_encodings(image_np, face_locations)
                
                if len(face_locations) == 0:
                    st.error("❌ Rasmdan hech qanday inson yuzi topilmadi.")
                else:
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
                        name = "Noma'lum"
                        
                        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = known_names[best_match_index]
                        
                        # Neon effekti bilan to'rtburchak chizish
                        cv2.rectangle(display_img, (left, top), (right, bottom), (0, 255, 127), 4)
                        cv2.rectangle(display_img, (left, top - 35), (right, top), (0, 255, 127), cv2.FILLED)
                        cv2.putText(display_img, name, (left + 6, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                    
                    st.markdown("### 📊 Skanerlash Natijasi:")
                    st.image(display_img, caption="AI Skanerlandi", use_container_width=True)
