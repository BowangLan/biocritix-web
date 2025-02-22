# !pip install requests beautifulsoup4 crewai crewai-tools openai

# from google.colab import userdata
import os

from dotenv import load_dotenv
from openai import OpenAI

dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env.local")
load_dotenv(dotenv_path)
# print(os.environ)
# Set up OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Could not find API key in environment variables")
    OPENAI_API_KEY = input("Please enter your OpenAI API key: ")
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is required to proceed")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

print("OpenAI API key has been set successfully!")

from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from crewai import Agent, Crew, Task
from crewai_tools import RagTool


class BiorxivSearcher:
    def __init__(self):
        # Update to use the details endpoint
        self.base_url = "https://api.biorxiv.org/details/biorxiv"

    def search_papers(self, keywords: List[str], max_results: int = 20) -> List[Dict]:
        print(f"keywords: {keywords}")
        all_papers = []
        for keyword in keywords:
            print(f"Searching for keyword: {keyword}")
            # Using the details endpoint with date range
            response = requests.get(
                f"{self.base_url}/2023-01-01/2024-12-31/0/{max_results}"
            )

            if response.status_code == 200:
                data = response.json()
                # Filter results that contain the keyword in title or abstract
                matching_papers = [
                    paper
                    for paper in data.get("collection", [])
                    if keyword.lower() in paper.get("title", "").lower()
                    or keyword.lower() in paper.get("abstract", "").lower()
                ]
                all_papers.extend(matching_papers)
            else:
                print(f"Failed response with code {response.status_code}")

        # Deduplicate papers based on DOI
        unique_papers = {paper["doi"]: paper for paper in all_papers}
        return list(unique_papers.values())[:max_results]

    def download_paper_html(self, doi: str) -> str:
        paper_url = f"https://www.biorxiv.org/content/{doi}.full"
        response = requests.get(paper_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.find("div", {"class": "main-content-wrapper"})
            print(content.get_text())
            return content.get_text() if content else ""
        return ""


def process_papers(papers: List[Dict]) -> List[str]:
    """Convert papers to text documents for RAG"""
    searcher = BiorxivSearcher()
    documents = []
    for paper in papers:
        content = searcher.download_paper_html(paper["doi"])
        if content:
            # Add metadata as header
            header = f"Title: {paper['title']}\nDOI: {paper['doi']}\n\n"
            documents.append(header + content)
    return documents


def main():
    # Interactive input version
    keywords = []
    print("Enter 4 search keywords (press Enter after each):")
    for i in range(4):
        keyword = input(f"Keyword {i+1}: ")
        keywords.append(keyword)

    # Initialize components
    searcher = BiorxivSearcher()
    print("post searcher")
    # Search and download papers
    papers = searcher.search_papers(keywords)
    print(f"Found {len(papers)} papers")

    if len(papers) == 0:
        print("test No papers found. Please try different keywords.")
        return

    # Process papers into documents
    documents = process_papers(papers)
    print("Papers processed and ready for analysis")

    # Create RAG tool
    rag_tool = RagTool(
        description="Search and analyze scientific papers from biorxiv",
        documents=documents,
    )

    # Create agents
    researcher = Agent(
        role="Research Analyst",
        goal="Analyze scientific papers and provide accurate insights",
        backstory="""You are an expert research analyst with deep knowledge of 
        scientific literature. You excel at identifying key findings and methodologies.""",
        tools=[rag_tool],
    )

    writer = Agent(
        role="Technical Writer",
        goal="Synthesize research findings into clear explanations",
        backstory="""You are a skilled technical writer who can explain complex 
        scientific concepts clearly and accurately. You excel at creating 
        comprehensive summaries.""",
        tools=[rag_tool],
    )

    # Create crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[
            Task(
                description="""Analyze the papers and identify key findings, methods, 
                and potential implications. Focus on extracting the most significant 
                research contributions.""",
                expected_output="""A detailed analysis of the key findings, methods, and implications 
                from the research papers, with specific examples and citations.""",
                agent=researcher,
            ),
            Task(
                description="""Create a clear, comprehensive summary of the research 
                findings. Explain complex concepts in an accessible way while maintaining 
                scientific accuracy.""",
                expected_output="""A clear, well-structured summary of the research findings that 
                explains complex concepts in an accessible way while maintaining accuracy.""",
                agent=writer,
            ),
        ],
    )

    # Run the crew
    result = crew.kickoff(max_iterations=12)
    print("\nCrewAI Analysis Results:")
    print(result)


if __name__ == "__main__":
    main()
