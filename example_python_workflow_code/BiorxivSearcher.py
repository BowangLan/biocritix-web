from lxml import html
from rich import print

from typing import Dict, List

import requests
from bs4 import BeautifulSoup


class BiorxivSearcher:
    base_url: str

    def __init__(self):
        # Update to use the details endpoint
        self.base_url = "https://biorxiv.org"

    def search_papers(self, keywords: List[str], page_size: int = 50) -> List[Dict]:
        print(f"Searching for keywords: {keywords}")

        response = requests.get(
            f"{self.base_url}/search/"
            + "+".join(keywords)
            + f" numresults:{page_size} sort:relevance-rank"
        )

        if response.status_code != 200:
            print(f"Failed response with code {response.status_code}")
            return []

        tree = html.fromstring(response.content)

        total_paper_count = tree.xpath(r"//h1[@id='page-title']/text()")
        if len(total_paper_count) == 0:
            print("No total paper count found")
            total_paper_count = 0
        else:
            try:
                total_paper_count = (
                    total_paper_count[0].strip("\n\t").replace(",", "").split(" ")[0]
                )
                total_paper_count = int(total_paper_count)
            except ValueError:
                print(f"Failed to parse total paper count: {total_paper_count}")
                total_paper_count = 0
        print(f"Found {total_paper_count} papers")

        results = []

        paper_items = tree.xpath(r'//*[@id="hw-advance-search-result"]/div/div/ul/li')
        for paper in paper_items:
            link = paper.xpath(r".//a[1]/@href")
            link = link[0] if len(link) > 0 else ""
            title = paper.xpath(r".//a[1]/span/text()")
            title = title[0].strip() if len(title) > 0 else ""

            results.append({"url": link, "title": title})

        return results

    def download_paper_html_2(self, url: str) -> str:
        print(f"Downloading paper content: {url}")
        response = requests.get(f"{self.base_url}{url}.full")
        if response.status_code == 200:
            tree = html.fromstring(response.content)
            content = tree.xpath(
                r'//*[@id="panels-ajax-tab-container-highwire_article_tabs"]/div[2]/div/div/div/div/div/div/div/div'
            )
            return content[0].text_content() if len(content) > 0 else ""
        return ""

    def download_paper_html(self, doi: str) -> str:
        paper_url = f"https://www.biorxiv.org/content/{doi}v1"
        response = requests.get(paper_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.find("div", {"class": "content"})
            return content.get_text() if content else ""
        return ""


def main():
    searcher = BiorxivSearcher()
    results = searcher.search_papers(["cancer", "cell"], 10)
    print(results)
    print(f"Found {len(results)} papers")
    for result in results:
        html = searcher.download_paper_html_2(result["url"])
        print(html)
        break


if __name__ == "__main__":
    main()
