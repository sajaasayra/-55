from kb import CHUNKS, INTENT, RARE, embed
from router import norm

CASES_ALL = ["first_issue", "renewal", "lost", "damaged"]


def retrieve_open(question, docs, want_fee):
    """لما الوثيقة معروفة والحالة مش معروفة: بحث بكل حالات الوثيقة."""
    qn = norm(question)
    qv = embed([question])[0]
    services = {f"{d}_{c}" for d in docs for c in CASES_ALL}
    cs = [c for c in CHUNKS if c["service"] in services]
    if want_fee:
        return [c for c in cs if c["field"] == "الرسوم"]

    def score(c):
        sc = float(c["vec"] @ qv)
        if any(norm(w) in qn for w in INTENT.get(c["field"], [])):
            sc += 1.0
        elif c["field"] in RARE:
            sc -= 1.0
        if c["field"] == "ملاحظات":
            sc += 0.6                     # الحقائق المخفية غالباً بالملاحظات
        return sc

    picked, seen = [], set()
    for c in sorted(cs, key=lambda c: -score(c)):
        body = c["text"].split(":\n", 1)[-1]
        if body in seen:
            continue                      # نفس النص موجود بخدمة ثانية
        seen.add(body)
        picked.append(c)
        if len(picked) == 4:
            break
    return picked