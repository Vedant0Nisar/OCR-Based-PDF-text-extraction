from pdf2image import convert_from_path
import pytesseract

# Path to your scanned PDF file
pdf_path = 'safepdfkit.pdf'

# Convert PDF pages to a list of PIL Image objects
pages = convert_from_path(pdf_path, 500) # 500 is a good DPI for high-res extraction

extracted_text = ""

# Iterate through each page image and run OCR
for page in pages:
    # Extract text from the image using Tesseract
    text = pytesseract.image_to_string(page)
    extracted_text += text + "\n"

# Print or save the extracted text
print(extracted_text)

with open('output.txt', 'w') as f:
    f.write(extracted_text)
