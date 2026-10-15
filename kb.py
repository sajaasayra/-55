import pathlib
import numpy as np
import ollama
import re          # <-- جديد

from router import norm

EMB = "bge-m3"
DATA = pathlib.Path(__file__).parent / "data"

FIELDS = ["الفئة المستفيدة", "أماكن تقديم الخدمة", "شروط تقديم الخدمة",
          "الوثائق المطلوبة للحصول على الخدمة", "إجراءات تقديم الخدمة",
          "الزمن المستغرق لتقديم الخدمة", "شركاء الخدمة", "ملاحظات", "الرسوم"]

NAMES = {
    "passport_first_issue": "جواز سفر لأول مرة",
    "passport_renewal": "تجديد جواز السفر",
    "passport_lost": "جواز سفر بدل فاقد",
    "passport_damaged": "جواز سفر بدل تالف",
    "family_book_first_issue": "دفتر عائلة لأول مرة",
    "family_book_renewal": "تجديد دفتر العائلة",
    "family_book_lost": "دفتر عائلة بدل فاقد",
    "family_book_damaged": "دفتر عائلة بدل تالف",
    "id_card_first_issue": "بطاقة ذكية لأول مرة",
    "id_card_renewal": "تجديد البطاقة الذكية",
    "id_card_lost": "بطاقة ذكية بدل فاقد",
    "id_card_damaged": "بطاقة ذكية بدل تالف",
}

CHUNKS = []   # كل عنصر: service, field, text, vec


def _embed_one(text):
    """نص واحد بكل مرة، على الـ CPU. إذا فشل (NaN) نقصّه ونعيد المحاولة."""
    for t in (text, text[:1000], text[:400]):
        try:
            r = ollama.embed(model=EMB, input=t, options={"num_gpu": 0})
            v = np.array(r["embeddings"][0], dtype=float)
            if np.isfinite(v).all() and np.linalg.norm(v) > 0:
                return v / np.linalg.norm(v)
        except Exception:
            pass
    return None


def embed(texts):
    vecs = [_embed_one(t) for t in texts]
    bad = sum(v is None for v in vecs)
    if bad:
        print(f"تحذير: فشل embedding لـ {bad} من {len(texts)} نص")
    dim = next((len(v) for v in vecs if v is not None), 1024)
    return np.array([v if v is not None else np.zeros(dim) for v in vecs])


def parse_file(path):
    """يرجّع {اسم_الحقل: نص}. أسطر القائمة بأول الملف بتنحذف لأنها قبل أول حقل."""
    fields, current = {}, None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        hit = next((h for h in FIELDS if s.startswith(h)), None)
        if hit:
            current = hit
            fields[current] = []
            s = s[len(hit):].strip()
        if current and s:
            fields[current].append(s)
    return {k: "\n".join(dict.fromkeys(v)) for k, v in fields.items()}

def load_chunks():
    for f in sorted(DATA.glob("*.txt")):
        for field, text in parse_file(f).items():
            if text.strip() in ("", "لا يوجد"):
                continue
            full = f"{NAMES.get(f.stem, f.stem)} - {field}:\n{text}"
            CHUNKS.append({"service": f.stem, "field": field, "text": full})
def build():
    load_chunks()
    vecs = embed([c["text"] for c in CHUNKS])
    for c, v in zip(CHUNKS, vecs):
        c["vec"] = v
    print(f"تم تحميل {len(CHUNKS)} chunk من {len(set(c['service'] for c in CHUNKS))} ملف")


# كلمات بتدل على الحقل اللي بده إياه المواطن (بتساعد حتى لو الـ embedding ضعيف)
INTENT = {
    "الوثائق المطلوبة للحصول على الخدمة": ["وثايق", "وثائق", "اوراق", "ورق", "مطلوب", "اجيب", "بلزم", "يلزم", "مستندات", "احضر", "صور"],
    "شروط تقديم الخدمة": ["شروط", "شرط", "يحق", "موافق", "لازم"],
    "الزمن المستغرق لتقديم الخدمة": ["يستغرق", "مده", "دقيق", "وقت", "ياخد", "بتاخد"],
    "أماكن تقديم الخدمة": ["وين", "اماكن", "مكان", "سفار", "قنصل", "مكتب"],
    "إجراءات تقديم الخدمة": ["خطوات", "اجراءات", "كيف", "طريقه"],
    "الفئة المستفيدة": ["مين", "فئه", "مستفيد"],
    "شركاء الخدمة": ["شركاء", "جهات", "جهه"],
}
RARE = ("إجراءات تقديم الخدمة", "شركاء الخدمة")   # بتنجاب فقط إذا السؤال بيطلبها


def retrieve(question, services, want_fee):
    """يرجّع الـ chunks من الخدمات المحددة فقط، وبأقل عدد لازم."""
    qn = norm(question)
    qv = embed([question])[0]
    picked = []
    for s in services:
        cs = [c for c in CHUNKS if c["service"] == s]
        must = [c for c in cs if want_fee and c["field"] in ("الرسوم", "ملاحظات")]

        def hit(c):
            return any(norm(w) in qn for w in INTENT.get(c["field"], []))

        def score(c):
            sc = float(c["vec"] @ qv)
            if hit(c):
                sc += 1.0
            elif c["field"] in RARE:
                sc -= 1.0                      # ما بنجيبها إلا إذا انطلبت
            if c["field"] == "ملاحظات":
                sc += 0.3                      # الحقائق المخفية غالباً بالملاحظات
            return sc

        rest = sorted((c for c in cs if c not in must), key=lambda c: -score(c))
        hits = sum(1 for c in rest if hit(c))
        if hits:
           n = 1 if len(services) > 1 else min(2, hits)              # السؤال واضح بنوع المعلومة
        elif want_fee:
            n = 0                              # الرسوم والملاحظات كافية
        else:
            n = 3 if len(services) == 1 else 2
        picked += must + rest[:n]
    return picked