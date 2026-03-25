import requests
from bs4 import BeautifulSoup
import json
import time
import io
from urllib.parse import urljoin, urlparse
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import pdfplumber
import os
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

START_URLS = {

    # ── MAIN PAGES ────────────────────────────────────────────────
    "home":                 "https://www.iitj.ac.in/",
    "about":                "https://www.iitj.ac.in/main/en/about-iitj",
    "announcements":        "https://www.iitj.ac.in/main/en/all-announcement",

    # ── FACULTY ───────────────────────────────────────────────────
    "faculty_all":          "https://www.iitj.ac.in/main/en/faculty-members",
    "faculty_adjunct":      "https://www.iitj.ac.in/main/en/adjunct-faculty-members",
    "faculty_visiting":     "https://www.iitj.ac.in/main/en/visiting-faculty-members",
    "faculty_scholars":     "https://www.iitj.ac.in/main/en/scholars-in-residence",

    # ── DEPARTMENTS (all 11) ──────────────────────────────────────
    "dept_bioscience":      "https://www.iitj.ac.in/bioscience-bioengineering/en/faculty-members",
    "dept_chemical":        "https://www.iitj.ac.in/chemical-engineering/en/faculty-members",
    "dept_chemistry":       "https://www.iitj.ac.in/chemistry/en/faculty-members",
    "dept_civil":           "https://www.iitj.ac.in/civil-infrastructure-engineering/en/faculty-members",
    "dept_cse":             "https://www.iitj.ac.in/computer-science-engineering/en/faculty-members",
    "dept_ee":              "https://www.iitj.ac.in/electrical-engineering/en/faculty-members",
    "dept_math":            "https://www.iitj.ac.in/mathematics/en/faculty-members",
    "dept_mech":            "https://www.iitj.ac.in/mechanical-engineering/en/faculty-members",
    "dept_meta":            "https://www.iitj.ac.in/metallurgical-materials-engineering/en/faculty-members",
    "dept_physics":         "https://www.iitj.ac.in/physics/en/faculty-members",

    # ── DEPARTMENT COURSES (syllabus) ─────────────────────────────
    "courses_bioscience":   "https://www.iitj.ac.in/bioscience-bioengineering/en/courses",
    "courses_chemical":     "https://www.iitj.ac.in/chemical-engineering/en/courses",
    "courses_chemistry":    "https://www.iitj.ac.in/chemistry/en/courses",
    "courses_civil":        "https://www.iitj.ac.in/civil-infrastructure-engineering/en/courses",
    "courses_cse":          "https://www.iitj.ac.in/computer-science-engineering/en/courses",
    "courses_ee":           "https://www.iitj.ac.in/electrical-engineering/en/courses",
    "courses_math":         "https://www.iitj.ac.in/mathematics/en/courses",
    "courses_mech":         "https://www.iitj.ac.in/mechanical-engineering/en/courses",
    "courses_meta":         "https://www.iitj.ac.in/metallurgical-materials-engineering/en/courses",
    "courses_physics":      "https://www.iitj.ac.in/physics/en/courses",

    # ── DEPARTMENT RESEARCH ───────────────────────────────────────
    "research_cse":         "https://www.iitj.ac.in/computer-science-engineering/en/research",
    "research_ee":          "https://www.iitj.ac.in/electrical-engineering/en/research",
    "research_mech":        "https://www.iitj.ac.in/mechanical-engineering/en/research",
    "research_physics":     "https://www.iitj.ac.in/physics/en/research",
    "research_chemistry":   "https://www.iitj.ac.in/chemistry/en/research",
    "research_office":      "https://www.iitj.ac.in/main/en/office-of-research-development",

    # ── SCHOOLS ───────────────────────────────────────────────────
    "school_liberal_arts":  "https://www.iitj.ac.in/school-of-liberal-arts",
    "school_management":    "https://www.iitj.ac.in/school-of-management-entrepreneurship/en/about",

    # ── ACADEMIC PROGRAMS ─────────────────────────────────────────
    "programs_list":        "https://iitj.ac.in/office-of-academics/en/list-of-academic-programs",
    "programs_ug":          "https://www.iitj.ac.in/main/en/ug-programs",
    "programs_pg":          "https://www.iitj.ac.in/main/en/pg-programs",
    "programs_phd":         "https://www.iitj.ac.in/main/en/phd-program",

    # ── ACADEMIC REGULATIONS (MUST) ───────────────────────────────
    "regulations":          "https://iitj.ac.in/office-of-academics/en/academic-regulations",
    "circulars":            "https://iitj.ac.in/office-of-academics/en/circulars",

    # ── NEWSLETTERS ───────────────────────────────────────────────
    "newsletter_techscape": "https://iitj.ac.in/techscape",
    "newsletter_mech":      "https://www.iitj.ac.in/mechanical-engineering/en/News-Newsletter",

    # ── ADMISSIONS ────────────────────────────────────────────────
    "admissions":           "https://www.iitj.ac.in/main/en/admissions",
    "admissions_pg":        "https://iitj.ac.in/office-of-academics/en/admission-to-postgraduate-programs",

}

scraped_data = []
visited_urls = set()
visited_pdfs = set()
all_clean_sentences = []
unique_sentences = set()

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_FILE = os.path.join(OUTPUT_DIR, "corpus.txt")
stop_words = set(stopwords.words('english'))

# Blacklist of web/HTML artifacts that appear as tokens but carry no meaning.
# These come from HTML element IDs, CSS class names, and CMS boilerplate text
# scraped from the IITJ website (e.g. "cid" = content-id used by the CMS).
WEB_ARTIFACTS = {
    "cid", "wim", "rti", "nccr", "cccd", "cert", "nirf",
    "nav", "btn", "div", "col", "row", "img", "src",
    "href", "html", "http", "www", "arrow",
    "downward", "upward", "intranet", "portal", "login",
    "redirecttologinpage", "copyright", "reserved",
    # Additional artifacts found from Word Cloud inspection
    "vksj", "issn", "obc", "gsa", "etc", "non",
}


def clean_and_tokenize(raw_text):
    """Clean raw text and return a lowercased, normalized string."""
    text = re.sub(r'http[s]?://\S+', ' ', raw_text)       # Remove URLs
    text = re.sub(r'\S+@\S+', ' ', text)                   # Remove emails
    # Keep only ASCII letters, digits and whitespace.
    # This also removes non-English text (Hindi/Devanagari etc.) since
    # those characters are non-ASCII — satisfying the assignment requirement.
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def process_and_store_text(raw_text):
    """Tokenize cleaned text and add unique sentences to the global corpus."""
    # Step 1: Split into sentences FIRST using original text (punctuation intact),
    # so that sent_tokenize can correctly detect sentence boundaries using periods.
    # Cleaning punctuation before this step would merge all text into one sentence.
    for sent in sent_tokenize(raw_text):
        # Step 2: Clean each sentence individually after splitting
        cleaned_sent = clean_and_tokenize(sent)
        tokens = [
            t for t in word_tokenize(cleaned_sent)
            if len(t) > 2
            and t not in stop_words
            and t not in WEB_ARTIFACTS      # remove HTML/CMS artifact tokens like "cid"
            and not t.isdigit()
        ]
        if len(tokens) >= 3:
            sent_str = " ".join(tokens)
            if sent_str not in unique_sentences:
                unique_sentences.add(sent_str)
                all_clean_sentences.append(tokens)


def is_valid_internal(url, base_domain):
    """Check if a URL belongs to the same domain and hasn't been visited."""
    parsed = urlparse(url)
    return parsed.netloc == base_domain and url not in visited_urls


def scrape_pdf(pdf_url):
    """
    Stream PDF from URL into memory (no file saved to disk) and extract text
    using pdfplumber — best accuracy for academic/multi-column PDFs.
    Returns extracted text string or empty string on failure.
    """
    if pdf_url in visited_pdfs:
        return ""
    visited_pdfs.add(pdf_url)

    try:
        print(f"  [PDF] Extracting: {pdf_url}")
        response = requests.get(pdf_url, timeout=20, verify=False)
        response.raise_for_status()

        text = ""
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "

        if text.strip():
            print(f"  [PDF] ✓ Extracted {len(text.split())} words")
        else:
            print(f"  [PDF] ⚠ No text found (possibly scanned/image PDF)")

        return text.strip()

    except Exception as e:
        print(f"  [PDF] ✗ Error: {e}")
        return ""


def scrape_iitj_site(max_pages=500):
    pages_to_crawl = list(START_URLS.values())
    base_domain = "www.iitj.ac.in"

    count = 0
    while pages_to_crawl and count < max_pages:
        url = pages_to_crawl.pop(0)
        if url in visited_urls:
            continue

        try:
            print(f"[{count+1}/{max_pages}] Scraping: {url}")
            response = requests.get(url, timeout=10, verify=False)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            visited_urls.add(url)

            # ── 1. Extract HTML Page Content ──────────────────────────────────
            title = soup.find('title').text.strip() if soup.find('title') else "No Title"
            content_div = soup.find('div', class_='content') or soup.find('main') or soup.body

            # Remove all non-content HTML tags — scripts, styles, navigation,
            # sidebars, forms and buttons are boilerplate and not meaningful text.
            for tag in content_div.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form', 'button']):
                tag.decompose()

            text_data = content_div.get_text(separator=' ', strip=True) if content_div else ""
            process_and_store_text(text_data)

            # ── 2. Extract & Process PDF / Doc Links ─────────────────────────
            doc_links = []
            for link in soup.find_all('a', href=True):
                href = urljoin(url, link['href'])

                # Handle PDF links — extract text directly into corpus
                if href.lower().endswith('.pdf'):
                    doc_entry = {"text": link.text.strip(), "url": href}
                    pdf_text = scrape_pdf(href)
                    if pdf_text:
                        process_and_store_text(pdf_text)
                        doc_entry["preview"] = pdf_text[:300] + "..."
                    doc_links.append(doc_entry)

                # Handle other doc types (no text extraction, just log)
                elif any(href.lower().endswith(ext) for ext in ['.doc', '.docx']):
                    doc_links.append({"text": link.text.strip(), "url": href})

                # ── 3. Discover new internal HTML pages ──────────────────────
                elif is_valid_internal(href, base_domain) and "iitj.ac.in" in href:
                    pages_to_crawl.append(href)

            scraped_data.append({
                "url": url,
                "title": title,
                "text": text_data[:1000] + "...",   # Truncated preview for JSON
                "documents": doc_links
            })

            count += 1
            time.sleep(0.1)

        except Exception as e:
            print(f"Error on {url}: {e}")

    # ── Save Outputs ──────────────────────────────────────────────────────────
    with open('iitj_data.json', 'w', encoding='utf-8') as f:
        json.dump(scraped_data, f, indent=4)

    with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
        for tokens in all_clean_sentences:
            f.write(' '.join(tokens) + '\n')

    print(f"\n{'='*60}")
    print(f"  Pages scraped     : {len(scraped_data)}")
    print(f"  PDFs processed    : {len(visited_pdfs)}")
    print(f"  Total sentences   : {len(all_clean_sentences)}")
    print(f"  Vocabulary size   : {len(set(t for s in all_clean_sentences for t in s))}")
    print(f"  Saved to          : iitj_data.json & corpus.txt")
    print(f"{'='*60}")

    generate_wordcloud()


def generate_wordcloud():
    """
    Reads the saved corpus.txt and generates a Word Cloud image
    from the most frequent words in the preprocessed corpus.
    """
    # Count word frequencies from all cleaned sentences
    all_tokens = [t for sent in all_clean_sentences for t in sent]
    freq = Counter(all_tokens)

    # Print dataset statistics
    print(f"\n{'='*50}")
    print(f"  Total Tokens    : {len(all_tokens)}")
    print(f"  Vocabulary Size : {len(freq)}")
    print(f"{'='*50}")
    print("\nTop 20 most frequent words:")
    for word, count in freq.most_common(20):
        print(f"  {word:<25} {count}")

    # Generate and save the Word Cloud
    wordcloud = WordCloud(
        width            = 1600,
        height           = 900,
        background_color = 'white',
        colormap         = 'Blues',
        max_words        = 200,
        min_font_size    = 10,
        max_font_size    = 150,
        collocations     = False,       # avoid repeating bigrams
        prefer_horizontal= 0.85,
    ).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Most Frequent Words in IIT Jodhpur Corpus',
                 fontsize=20, fontweight='bold', pad=20)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "wordcloud.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nWord Cloud saved to: {output_path}")


if __name__ == "__main__":
    scrape_iitj_site(max_pages=1000)