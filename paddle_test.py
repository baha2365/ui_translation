from paddleocr import PaddleOCR

# 1. Модельді инициализациялау
ocr = PaddleOCR(use_textline_orientation=True, lang='ru')

img_path = 'Screenshot 2026-06-29 122416.png'

# 2. Мәтінді тану (жаңа predict функциясы)
results = ocr.predict(img_path)

print("\n--- ТАБЫЛҒАН МӘТІНДЕР ---")

# 3. Нәтижені қауіпсіз шығару
for res in results:
    # Егер нәтиже сөздік (dict) форматында келсе
    if isinstance(res, dict):
        texts = res.get('rec_texts', [])
        scores = res.get('rec_scores', [])
        for text, score in zip(texts, scores):
            print(f"Мәтін: '{text}' | Сенімділік: {float(score):.2f}")
    
    # Егер нәтиже тізім (list) форматында келсе
    elif isinstance(res, list):
        for line in res:
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                # Егер line[1] кортеж болса: (text, score)
                if isinstance(line[1], (list, tuple)):
                    text, score = line[1][0], line[1][1]
                # Егер line[1] тікелей мәтін сөзі болса
                else:
                    text = line[1]
                    score = line[2] if len(line) > 2 else 1.0
                print(f"Мәтін: '{text}' | Сенімділік: {float(score):.2f}")