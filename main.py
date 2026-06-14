import os
import cv2
import numpy as np
from pathlib import Path
from numpy.linalg import norm
# TensorFlow loglari kamaytiramiz (MTCNN uchun)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from mtcnn import MTCNN
from insightface.app import FaceAnalysis
from insightface.utils import face_align

THRESHOLD   = 0.50      # cosine o'xshashlik chegarasi (0.35-0.50 orasida)
DB_DIR      = Path("embeddings")
IMG_SIZE    = 112       # ArcFace input o'lchami

DB_DIR.mkdir(exist_ok=True)

print("[INFO] MTCNN yuklanmoqda...")
detector = MTCNN()

print("[INFO] ArcFace yuklanmoqda (birinchi marta ~200MB yuklanadi)...")
# Barcha modullar bilan yuklaymiz (detection talab qilinadi)
# Lekin detection uchun MTCNN ishlatamiz - InsightFace detektorini ishlatmaymiz
_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
_app.prepare(ctx_id=0, det_size=(640, 640))

# Faqat ArcFace recognition modelini ajratib olamiz
arcface = _app.models["recognition"]
print("[INFO] Modellar tayyor!\n")
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Ikki vektor orasidagi cosine o'xshashlik [-1, 1]."""
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-8))
def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (norm(v) + 1e-8)
def mtcnn_landmarks_5(keypoints: dict) -> np.ndarray:
    """
    MTCNN keypoints -> ArcFace 5-nuqtali landmark formati.
    ArcFace tartibi: left_eye, right_eye, nose, mouth_left, mouth_right
    """
    order = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
    return np.array([keypoints[k] for k in order], dtype=np.float32)
def extract_embedding(bgr_frame: np.ndarray, box: list, keypoints: dict):
    """
    MTCNN natijasidan ArcFace embedding olish.

    Jarayon:
      1. MTCNN landmarklarini ArcFace formatiga o'tkazish
      2. face_align.norm_crop -> 112x112 tekislangan yuz
      3. ArcFace ONNX modeli -> 512-o'lchamli vektor
      4. L2 normalizatsiya
    """
    try:
        x, y, w, h = box
        x, y = max(0, x), max(0, y)      # manfiy koordinatalardan himoya

        lmk = mtcnn_landmarks_5(keypoints)

        # ArcFace RGB talab qiladi
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # 112x112 o'lchamga tekislangan crop
        aligned = face_align.norm_crop(rgb, landmark=lmk, image_size=IMG_SIZE)

        # get_feat() ba'zi versiyalarda ishlamaydi - xavfsiz usul
        try:
            emb = arcface.get_feat(aligned).flatten()
        except (AttributeError, Exception):
            inp = aligned.astype('float32').transpose(2,0,1)[None]
            inp = (inp - 127.5) / 127.5
            session = arcface.session
            iname = session.get_inputs()[0].name
            emb = session.run(None, {iname: inp})[0].flatten()
        return l2_normalize(emb)
    except Exception as e:
        print(f"  [XATO] Embedding: {e}")
        return None
def load_database() -> dict:
    """Barcha saqlangan embeddinglarni yuklash."""
    db = {}
    for f in DB_DIR.glob("*.npy"):
        db[f.stem] = np.load(f)
    return db
def find_best_match(embedding: np.ndarray, db: dict):
    """
    DB dagi eng yaqin foydalanuvchini topish.
    Qaytaradi: (user_id, similarity) yoki ("Noma'lum", max_sim)
    """
    best_id  = "Noma'lum"
    best_sim = -1.0
    for uid, stored in db.items():
        sim = cosine_sim(embedding, stored)
        if sim > best_sim:
            best_sim = sim
            if sim >= THRESHOLD:
                best_id = uid

    return best_id, best_sim
class Camera:
    """Kamerani boshqarish uchun context manager."""

    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index)
    def __enter__(self):
        if not self.cap.isOpened():
            print("[XATO] Kamera topilmadi!")
            sys.exit(1)
        return self.cap
    def __exit__(self, *_):
        self.cap.release()
        cv2.destroyAllWindows()
def capture_face(title: str = "Yuz olish"):
    """
    Kamera oynasini ochadi, MTCNN bilan yuzni aniqlaydi.
    SPACE  -> embedding olib qaytaradi
    Q      -> None qaytaradi
    """
    with Camera() as cap:
        print(f"  [{title}]  SPACE=olish   Q=chiqish")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("  [XATO] Kadr o'qilmadi.")
                break
            display = frame.copy()
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # MTCNN detection
            detections = detector.detect_faces(rgb)
            for det in detections:
                x, y, w, h = det["box"]
                x, y = max(0, x), max(0, y)
                conf  = det["confidence"]
                # yashil=ishonchli, sariq=shubhali
                color = (0, 220, 0) if conf > 0.95 else (0, 200, 220)
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display, f"{conf:.2f}",
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color, 1)
                # 5 ta landmark nuqta
                if "keypoints" in det:
                    for pt in det["keypoints"].values():
                        cv2.circle(display, (int(pt[0]), int(pt[1])),
                                   3, (255, 120, 0), -1)
            n = len(detections)
            status = f"Yuzlar: {n}" if n else "Yuz topilmadi"
            cv2.putText(display, status,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
            cv2.putText(display, "SPACE=olish  Q=chiqish",
                        (10, display.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (180, 180, 180), 1)
            cv2.imshow(title, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if n == 0:
                    print("  [!] Yuz topilmadi. Kameraga yaqinroq turing.")
                    continue
                if n > 1:
                    print("  [!] Bir nechta yuz bor. Faqat bitta yuz ko'rsating.")
                    continue
                emb = extract_embedding(frame,
                                        detections[0]["box"],
                                        detections[0]["keypoints"])
                if emb is not None:
                    print("  [OK] Yuz muvaffaqiyatli olindi.")
                    return emb
            elif key == ord("q"):
                print("  [!] Bekor qilindi.")
                return None
def realtime_recognition():
    """
    Real vaqtda barcha ro'yxatdagi foydalanuvchilarni avtomatik taniydi.
    ID kiritish shart emas.

    Optimallashtirish:
      - Har 3-kadrda bir marta MTCNN detection (CPU yuki kamaytirish)
      - Embedding faqat ishonchli yuzdangina (conf > 0.90)
      - Oxirgi natija keyingi kadrda ham ko'rsatiladi (lag kamaytirish)
    """
    db = load_database()
    if not db:
        print("[!] Baza bo'sh. Avval ro'yxatdan o'ting (1-band).")
        return

    print(f"[INFO] Baza: {len(db)} ta foydalanuvchi -> {list(db.keys())}")
    print("  Real vaqtli tanish boshlandi.  Q = chiqish\n")

    SKIP_FRAMES = 3
    frame_cnt   = 0
    last_dets   = []

    with Camera() as cap:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Har SKIP_FRAMES kadrda detection
            if frame_cnt % SKIP_FRAMES == 0:
                last_dets = detector.detect_faces(rgb)
            frame_cnt += 1

            for det in last_dets:
                x, y, w, h = det["box"]
                x, y = max(0, x), max(0, y)
                conf  = det["confidence"]

                label     = "Noma'lum"
                box_color = (0, 0, 200)      # qizil

                if conf > 0.90 and "keypoints" in det:
                    emb = extract_embedding(frame, det["box"], det["keypoints"])
                    if emb is not None:
                        uid, sim = find_best_match(emb, db)
                        if uid != "Noma'lum":
                            label     = f"{uid}  {sim:.2f}"
                            box_color = (0, 200, 0)    # yashil
                        else:
                            label = f"Noma'lum  {sim:.2f}"

                cv2.rectangle(display, (x, y), (x+w, y+h), box_color, 2)
                cv2.putText(display, label,
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, box_color, 2)

            cv2.putText(display, f"DB: {len(db)} kishi  |  Q=chiqish",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

            cv2.imshow("Real-Time Yuz Tanish", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
def register():
    print("\n========= RO'YXATDAN O'TISH =========")
    uid = input("  Foydalanuvchi ID: ").strip()
    if not uid:
        print("  [!] ID bo'sh bo'lmasligi kerak.")
        return

    path = DB_DIR / f"{uid}.npy"
    if path.exists():
        ans = input(f"  '{uid}' allaqachon bor. Qayta yozilsinmi? (y/n): ")
        if ans.lower() != "y":
            return

    emb = capture_face(title=f"Ro'yxat: {uid}")
    if emb is None:
        print("  [!] Yuz olinmadi.")
        return

    np.save(path, emb)
    print(f"  [OK] '{uid}' saqlandi -> {path}")
def verify_once():
    print("\n========= TEKSHIRUV =========")
    uid  = input("  ID kiriting: ").strip()
    path = DB_DIR / f"{uid}.npy"

    if not path.exists():
        print(f"  [!] '{uid}' bazada topilmadi.")
        return

    stored = np.load(path)
    emb    = capture_face(title=f"Tekshiruv: {uid}")

    if emb is None:
        print("  [!] Yuz olinmadi.")
        return

    sim = cosine_sim(emb, stored)
    print(f"\n  O'xshashlik : {sim:.4f}")
    print(f"  Chegara     : {THRESHOLD}")

    if sim >= THRESHOLD:
        print("  [+] KIRISH RUXSAT ETILDI")
    else:
        print("  [-] KIRISH RAD ETILDI")
def list_users():
    users = sorted(p.stem for p in DB_DIR.glob("*.npy"))
    if not users:
        print("  [!] Baza bo'sh.")
    else:
        print(f"\n  Ro'yxatdagi foydalanuvchilar ({len(users)} ta):")
        for u in users:
            print(f"    - {u}")
def delete_user():
    list_users()
    uid  = input("\n  O'chirish uchun ID: ").strip()
    path = DB_DIR / f"{uid}.npy"
    if path.exists():
        path.unlink()
        print(f"  [OK] '{uid}' o'chirildi.")
    else:
        print(f"  [!] '{uid}' topilmadi.")
MENU = """
+======================================+
|   MTCNN + ArcFace  Yuz Tanish        |
+======================================+
|  1.  Ro'yxatdan o'tish               |
|  2.  Bir martalik tekshiruv (ID li)  |
|  3.  Real vaqtli tanish (avtomatik)  |
|  4.  Foydalanuvchilar ro'yxati       |
|  5.  Foydalanuvchi o'chirish         |
|  6.  Chiqish                         |
+======================================+"""
def main():
    actions = {
        "1": register,
        "2": verify_once,
        "3": realtime_recognition,
        "4": list_users,
        "5": delete_user,
    }

    while True:
        print(MENU)
        choice = input("  Tanlov: ").strip()
        if choice == "6":
            print("  Xayr!")
            break
        action = actions.get(choice)
        if action:
            action()
        else:
            print("  [!] Noto'g'ri tanlov, qayta kiriting.")
if __name__ == "__main__":
    main()
