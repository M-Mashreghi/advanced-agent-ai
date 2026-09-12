import pymupdf


def extract_pages(uploaded_file):
    data = uploaded_file.getvalue()
    doc = pymupdf.open(stream=data, filetype="pdf")
    pages = []
    for page_no, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"filename": uploaded_file.name, "page": page_no, "text": text})
    doc.close()
    return pages
