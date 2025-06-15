import pdfplumber
with pdfplumber.open('MakeYour HomeSafe.pdf') as pdf:
    with open('makeyourhome_safe_excerpt.txt', 'w', encoding='utf-8') as f:
        for page in pdf.pages[:5]:
            text = page.extract_text()
            if text:
                f.write(text + '\n')