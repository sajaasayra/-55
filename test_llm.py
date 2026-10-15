import ollama, time

for opts in ({"num_ctx": 2048}, {"num_ctx": 2048, "num_gpu": 0}):
    t = time.time()
    try:
        r = ollama.chat(
            model="command-r7b-arabic:7b",
            messages=[{"role": "user", "content": "كم رسوم تجديد الجواز؟ جاوب بكلمة"}],
            options=opts,
        )
        print("OK", opts, round(time.time() - t), "ثانية:", r["message"]["content"][:80])
    except Exception as e:
        print("FAIL", opts, e)