from pypdf import PdfReader


def extract_text(file_path: str):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text