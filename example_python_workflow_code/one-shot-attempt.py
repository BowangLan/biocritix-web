import requests
from bs4 import BeautifulSoup
import argparse
from typing import List, Dict
from crewai import Agent, Task, Crew
from crewai_tools import RAGTool

class BiorxivSearcher:
    def __init__(self):
        self.base_url = "https://api.biorxiv.org/details/biorxiv"
        
    def search_papers(self, keywords: List[str], max_results: int = 5) -> List[Dict]:
        all_papers = []
        for keyword in keywords:
            response = requests.get(f"{self.base_url}/{keyword}/0/5")
            if response.status_code == 200:
                data = response.json()
                all_papers.extend(data.get('collection', []))
        
        # Deduplicate papers based on DOI
        unique_papers = {paper['doi']: paper for paper in all_papers}
        return list(unique_papers.values())[:max_results]

    def download_paper_html(self, doi: str) -> str:
        paper_url = f"https://www.biorxiv.org/content/{doi}v1"
        response = requests.get(paper_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.find('div', {'class': 'content'})
            return content.get_text() if content else ""
        return ""

def process_papers(papers: List[Dict]) -> List[str]:
    """Convert papers to text documents for RAG"""
    searcher = BiorxivSearcher()
    documents = []
    for paper in papers:
        content = searcher.download_paper_html(paper['doi'])
        if content:
            # Add metadata as header
            header = f"Title: {paper['title']}\nDOI: {paper['doi']}\n\n"
            documents.append(header + content)
    return documents

def main():
    parser = argparse.ArgumentParser(description='Search and analyze biorxiv papers')
    parser.add_argument('--prompts', nargs=4, required=True, help='Four search keywords')
    args = parser.parse_args()

    # Initialize components
    searcher = BiorxivSearcher()
    
    # Search and download papers
    papers = searcher.search_papers(args.prompts)
    print(f"Found {len(papers)} papers")
    
    # Process papers into documents
    documents = process_papers(papers)
    print("Papers processed and ready for analysis")
    
    # Create RAG tool
    rag_tool = RAGTool(
        description="Search and analyze scientific papers from biorxiv",
        documents=documents
    )
    
    # Create agents
    researcher = Agent(
        role='Research Analyst',
        goal='Analyze scientific papers and provide accurate insights',
        backstory="""You are an expert research analyst with deep knowledge of 
        scientific literature. You excel at identifying key findings and methodologies.""",
        tools=[rag_tool]
    )
    
    writer = Agent(
        role='Technical Writer',
        goal='Synthesize research findings into clear explanations',
        backstory="""You are a skilled technical writer who can explain complex 
        scientific concepts clearly and accurately. You excel at creating 
        comprehensive summaries.""",
        tools=[rag_tool]
    )
    
    # Create crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[
            Task(
                description="""Analyze the papers and identify key findings, methods, 
                and potential implications. Focus on extracting the most significant 
                research contributions.""",
                agent=researcher
            ),
            Task(
                description="""Create a clear, comprehensive summary of the research 
                findings. Explain complex concepts in an accessible way while maintaining 
                scientific accuracy.""",
                agent=writer
            )
        ]
    )
    
    # Run the crew
    result = crew.kickoff()
    print("\nCrewAI Analysis Results:")
    print(result)

if __name__ == "__main__":
    main()

# Requirements.txt
"""
requests
beautifulsoup4
crewai
crewai-tools
"""