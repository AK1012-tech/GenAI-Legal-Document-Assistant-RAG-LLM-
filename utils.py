from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def extract_text_from_url(url):
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')
    return ' '.join(p.get_text() for p in soup.find_all('p'))

'''
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup

def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        print(f"[ERROR] Failed to extract text from PDF: {e}")
        return "Error extracting text from the PDF."

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Optionally extract title and headings
        title = soup.title.string.strip() if soup.title else ""
        headings = ' '.join(h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3']))
        paragraphs = ' '.join(p.get_text(strip=True) for p in soup.find_all('p'))

        full_text = "\n".join([title, headings, paragraphs]).strip()
        return full_text if full_text else "No extractable content found on the page."
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to retrieve URL content: {e}")
        return "Error retrieving content from the URL."
    except Exception as e:
        print(f"[ERROR] Failed to parse HTML content: {e}")
        return "Error parsing HTML content from the URL."
'''
