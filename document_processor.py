import pdfplumber

def extract_pdf_chunks(r"C:\Users\HP\bajaj\bajaj_finserv_factsheet_Oct.pdf", chunk_size=1024):
    chunks = []
    with pdfplumber.open(r"C:\Users\HP\bajaj\bajaj_finserv_factsheet_Oct.pdf") as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            while len(text) > chunk_size:
                chunks.append(text[:chunk_size])
                text = text[chunk_size:]
            if text.strip():
                chunks.append(text)
    return chunks
