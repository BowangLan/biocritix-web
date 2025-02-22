import requests
import json
import lancedb
import os
from bs4 import BeautifulSoup
import argparse
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from crewai import Agent, Task, Crew
from langchain.tools import Tool
from langchain.agents import Tool
from langchain_openai import ChatOpenAI

class BiorxivSearcher:
    def __init__(self):
        self.base_url = "https://api.biorxiv.org/details/biorxiv"
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
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
        # Convert DOI to biorxiv URL
        paper_url = f"https://www.biorxiv.org/content/{doi}v1"
        response = requests.get(paper_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract main content
            content = soup.find('div', {'class': 'content'})
            return content.get_text() if content else ""
        return ""

class VectorStore:
    def __init__(self):
        self.db = lancedb.connect('~/paper_store.lance')
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def create_table(self):
        schema = {
            "text": "string",
            "doi": "string",
            "title": "string",
            "vector": "float32[384]"  # Dimension matches all-MiniLM-L6-v2
        }
        if "papers" not in self.db.table_names():
            self.db.create_table("papers", schema=schema)
        return self.db.open_table("papers")

    def add_papers(self, papers: List[Dict]):
        table = self.create_table()
        for paper in papers:
            html_content = BiorxivSearcher().download_paper_html(paper['doi'])
            # Split content into chunks (simplified chunking)
            chunks = [html_content[i:i+1000] for i in range(0, len(html_content), 1000)]
            
            for chunk in chunks:
                vector = self.encoder.encode(chunk)
                table.add([{
                    "text": chunk,
                    "doi": paper['doi'],
                    "title": paper['title'],
                    "vector": vector.tolist()
                }])

    def search(self, query: str, k: int = 3):
        table = self.db.open_table("papers")
        query_vector = self.encoder.encode(query)
        results = table.search(query_vector).limit(k).to_list()
        return results

class PaperAnalysisAgent:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")
        
    def create_researcher(self) -> Agent:
        return Agent(
            role='Research Analyst',
            goal='Analyze scientific papers and provide accurate insights',
            backstory='You are an expert research analyst with deep knowledge of scientific literature',
            llm=self.llm,
            tools=[
                Tool(
                    name='Search Papers',
                    func=self.search_papers,
                    description='Search through the paper database'
                )
            ]
        )
    
    def create_writer(self) -> Agent:
        return Agent(
            role='Technical Writer',
            goal='Synthesize research findings into clear explanations',
            backstory='You are a skilled technical writer who can explain complex concepts clearly',
            llm=self.llm,
            tools=[
                Tool(
                    name='Search Papers',
                    func=self.search_papers,
                    description='Search through the paper database'
                )
            ]
        )
    
    def search_papers(self, query: str) -> str:
        results = self.vector_store.search(query)
        return json.dumps([{
            'title': r['title'],
            'text': r['text'][:500] + '...' if len(r['text']) > 500 else r['text']
        } for r in results])

def main():
    parser = argparse.ArgumentParser(description='Search and analyze biorxiv papers')
    parser.add_argument('--prompts', nargs=4, required=True, help='Four search keywords')
    args = parser.parse_args()

    # Initialize components
    searcher = BiorxivSearcher()
    vector_store = VectorStore()
    
    # Search and download papers
    papers = searcher.search_papers(args.prompts)
    print(f"Found {len(papers)} papers")
    
    # Store papers in vector database
    vector_store.add_papers(papers)
    print("Papers stored in vector database")
    
    # Initialize agents
    agent_system = PaperAnalysisAgent(vector_store)
    researcher = agent_system.create_researcher()
    writer = agent_system.create_writer()
    
    # Create crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[
            Task(
                description="Analyze the papers and identify key findings",
                agent=researcher
            ),
            Task(
                description="Create a clear summary of the research findings",
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
lancedb
sentence-transformers
crewai
langchain
langchain-openai
"""