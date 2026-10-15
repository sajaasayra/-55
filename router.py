import re


def norm(s):
    s = re.sub("[\u064B-\u0652]", "", s)           # التشكيل
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه")):
        s = s.replace(a, b)
    return s


DOCS = {
    "passport": ["جواز", "باسبور", "سفر"],
      "family_book": ["دفتر", "عائل", "عايل", "عيل", "رب الاسر"],
    "id_card": ["بطاق", "هوي"],
}
CASES = {
    "first_issue": ["اول مره", "لاول"],
    "renewal": ["جدد", "تجديد", "منتهي", "انتهت", "انتهي", "خلص"],
    "lost": ["فاقد", "ضاع", "ضيع", "فقد", "مفقود", "ضايع"],
    "damaged": ["تالف", "تلف", "خربان", "خرب", "ممزق", "متضرر"],
}
FEE_WORDS = ["رسوم", "كلف", "سعر", "بدفع", "دفع", "غرام", "كفال",
             "تعهد", "اغلي", "ارخص", "مبلغ", "فلوس", "استعجل", "مستعجل", "اضافي"]


def find(table, q):
    q = norm(q)
    return [k for k, words in table.items() if any(norm(w) in q for w in words)]


def wants_fee(q):
    q = norm(q)
    return any(norm(w) in q for w in FEE_WORDS)


WEAK_FIRST = ["اصدار", "اصدر", "استخراج", "اطلع"]   # تعني "أول مرة" فقط إذا ما ذُكرت حالة ثانية


def find_cases(q):
    res = find(CASES, q)
    if not res and any(norm(w) in norm(q) for w in WEAK_FIRST):
        res = ["first_issue"]
    return res


def _spans(q, words):
    """مواضع (بداية، نهاية) لكل كلمة من القائمة موجودة بالسؤال."""
    q = norm(q)
    out = []
    for w in words:
        w = norm(w)
        p = q.find(w)
        if p >= 0:
            out.append((p, p + len(w)))
    return out


def _gap(a, b):
    """أقل مسافة بين مجموعتي مواضع."""
    return min(max(0, max(s1, s2) - min(e1, e2)) for s1, e1 in a for s2, e2 in b)


def pair_services(q, docs, cases):
    """يربط كل حالة بأقرب وثيقة لها بالسؤال، بدل ضرب كل الوثائق بكل الحالات."""
    all_pairs = [f"{d}_{c}" for d in docs for c in cases]
    if len(docs) == 1 or len(cases) == 1:
        return all_pairs
    dsp = {d: _spans(q, DOCS[d]) for d in docs}
    csp = {c: _spans(q, CASES[c]) for c in cases}
    if any(not v for v in dsp.values()) or any(not v for v in csp.values()):
        return all_pairs                       # (جاية من ذاكرة المحادثة)
    assigned = {d: [] for d in docs}
    for c in cases:
        assigned[min(docs, key=lambda d: _gap(dsp[d], csp[c]))].append(c)
    return [f"{d}_{c}" for d in docs for c in (assigned[d] or cases)]