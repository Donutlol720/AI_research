import arxiv

#https://arxiv.org/pdf/2408.04667

import requests
from bs4 import BeautifulSoup
import re
import os

def sanitize_filename(s: str, max_len: int = 200) -> str:
    s = re.sub(r'[\\/:"*?<>|]+', '', s)
    s = s.strip().replace(' ', '_')
    return s[:max_len]

def fetch_title_from_abs(abs_url: str) -> str:
    resp = requests.get(abs_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    ld = soup.find("script", type="application/ld+json")
    if ld:
        import json
        data = json.loads(ld.string)
        return data.get("name")
    title_tag = soup.find('h1', class_='title')
    if title_tag:
        return title_tag.get_text(strip=True).replace('Title:', '')
    return None

def download_file(url: str, filename: str):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def process_link(url: str, out_dir: str = "."):
    print(f"\nProcessing {url}")
    # Determine base arXiv ID if arXiv link
    if "arxiv.org" in url:
        arxiv_id = None
        match = re.search(r'arxiv\.org/(?:pdf|html|abs)/([^/]+)', url)
        if match:
            arxiv_id = match.group(1).replace('.pdf','').replace('v1','').split('v')[0]
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        title = fetch_title_from_abs(abs_url)
        if not title:
            title = arxiv_id
        safe = sanitize_filename(title)
        filename = os.path.join(out_dir, safe + ".pdf")
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        download_file(pdf_url, filename)
        print(f"Saved as: {filename}")
    else:
        # Non-arXiv links (html/PDF)
        title = None
        if "aclweb.org" in url or "aclanthology.org" in url:
            # Could add scraping logic for ACL pages
            title = None
        safe = sanitize_filename(title or os.path.basename(url).split('?')[0])
        filename = os.path.join(out_dir, safe + ".pdf")
        download_file(url, filename)
        print(f"Downloaded file as: {filename}")

if __name__ == "__main__":
    links = [
        "https://arxiv.org/pdf/2303.08774",
        "https://arxiv.org/pdf/2201.11903",
        "https://arxiv.org/pdf/2402.00157",
        "https://arxiv.org/abs/2402.12091",
        "https://arxiv.org/html/2502.03671",
        "https://arxiv.org/pdf/2412.09078v5",
        "https://arxiv.org/pdf/2210.03629",
        "https://aclanthology.org/2023.acl-long.153.pdf",
        "https://arxiv.org/pdf/2401.05618",
        "https://arxiv.org/html/2503.01933v1",
        "https://arxiv.org/abs/2506.02153",
        "https://arxiv.org/html/2503.06519v1",
        "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
        "https://arxiv.org/pdf/2505.17652",
        "https://arxiv.org/pdf/2408.04667",
        "https://arxiv.org/pdf/2504.01282",
        "https://arxiv.org/pdf/2205.11916",
        "https://aclanthology.org/2023.findings-emnlp.123.pdf",
        "https://arxiv.org/pdf/2302.00093",
        "https://arxiv.org/pdf/2402.14848",
        "https://arxiv.org/pdf/2505.00019",
        "https://arxiv.org/pdf/2505.17037",
        "https://arxiv.org/pdf/2406.10248",
        "https://arxiv.org/pdf/2506.08669",
        "https://arxiv.org/pdf/2504.09923",
        "https://arxiv.org/pdf/2408.06195",
        "https://arxiv.org/pdf/2403.09606",
    ]
    for link in links:
        process_link(link, out_dir="ignore_ai_paper")





# search = arxiv.Search(id_list=["2408.04667"])
# paper = next(search.results())
# paper.download_pdf(filename="2502.03671.pdf")

# import urllib.request

# urllib.request.urlretrieve(
#     "https://arxiv.org/pdf/2408.04667",
#     "2408.04667.pdf"
# )


# import PyPDF2

# def extract_title_author(pdf_path):
#     with open(pdf_path, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         first_page = reader.pages[0]
#         text = first_page.extract_text()
        
#         # Simple parsing - you may need to adjust based on PDF format
#         lines = text.split('\n')
#         #print(lines)
#         title = lines[0] if lines else "Title not found"
#         author = lines[1] if len(lines) > 1 else "Author not found"
        
#         return title, author

# # Usage
# title, author = extract_title_author('2502.03671.pdf')
# print(f"Title: {title}")
# print(f"Author: {author}")



# import json
# import matplotlib.pyplot as plt

# def analyze_results(file_path):
#     with open(file_path, 'r') as f:
#         data = json.load(f)
    
#     problem_keys = []
#     results = []
#     error_flag_count = 0

#     for problem_key, problem_data in data.items():
#         problem_keys.append(problem_key)
#         results.append(problem_data['result'])
#         if problem_data['flag_error'] == 1:
#             error_flag_count += 1

#     return problem_keys, results, error_flag_count

# def visualize_results(problem_keys, results):
#     # Convert results to numerical values for visualization
#     correct_counts = [int(result.split('/')[0]) for result in results]
#     total_counts = [int(result.split('/')[1]) for result in results]

#     # Create a bar plot
#     plt.figure(figsize=(10, 6))
#     plt.bar(problem_keys, correct_counts, label='Correct Answers')
#     plt.bar(problem_keys, total_counts, alpha=0.5, label='Total Attempts')

#     plt.xlabel('Problem Keys')
#     plt.ylabel('Count')
#     plt.title('Results Visualization')
#     plt.xticks(rotation=90)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     file_path = r'.\Results\AMC_2022_12A_llama-4-maverick_benchmark_Results.json'
#     problem_keys, results, error_flag_count = analyze_results(file_path)

#     print("Problem Keys and Results:")
#     for key, result in zip(problem_keys, results):
#         print(f"{key}: {result}")

#     print(f"\nFrequency of Error Flag: {error_flag_count}")

#     # Visualize the results
#     visualize_results(problem_keys, results)