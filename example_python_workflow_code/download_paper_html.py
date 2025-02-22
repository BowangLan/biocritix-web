from bs4 import BeautifulSoup


def download_paper_html(self, doi: str) -> str:
    paper_url = f"https://www.biorxiv.org/content/{doi}.full"
    response = requests.get(paper_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("div", {"class": "main-content-wrapper"})
        print(content.get_text())
        return content.get_text() if content else ""
    return ""
