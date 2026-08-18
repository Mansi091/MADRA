import uuid
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from ddgs import DDGS
from langchain_groq import ChatGroq
import time
import random
import diskcache as dc
import asyncio
from state import ResearchState


cache = dc.Cache(".research_cache", size_limit=50 * 1024 * 1024)

class Claim(BaseModel):
    claim: str = Field(description="A distinct factual assertion.")

class EvidenceExtraction(BaseModel):
    claims: list[Claim]

def researcher_node(state: dict) -> dict:
    sub_question = state.get("sub_question")
    print(f"\n[AGENT 2: RESEARCHER] RUNNING FOR: {sub_question}")

    print("Searching web")
    try:
        search_results = DDGS().text(sub_question, max_results=2)
    except Exception as e:
        print(f"Search failed: {e}")
        search_results = []
    
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    claim_llm = llm.with_structured_output(EvidenceExtraction)
    
    evidence_list = []
    urls_to_check = [res.get("href") for res in search_results if res.get("href")]

    evidence_list = asyncio.run(process_all_urls(urls_to_check, sub_question, claim_llm))
            
    print(f"Found {len(evidence_list)} sources with verified claims.")
    return {"evidence": evidence_list}

async def process_all_urls(urls, sub_question, claim_llm):
    tasks = [process_url(url, sub_question, claim_llm) for url in urls]
    results = await asyncio.gather(*tasks)
    return [res for res in results if res is not None]

async def process_url(url, sub_question, claim_llm):
    cache_key = f"{sub_question}::{url}"
    
    if cache_key in cache:
        print(f"[CACHE HIT] Loaded extracted claims for: {url}")
        cached_claims = cache[cache_key]
        if cached_claims:
            return {
                "source_id": str(uuid.uuid4())[:8],
                "question": sub_question,
                "url": url,
                "claims": cached_claims
            }
        return None

    try:
        print(f"Reading source: {url}")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = await asyncio.to_thread(requests.get, url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.content, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text(separator=' ', strip=True)[:6000]
        
        if len(text) < 100: 
            cache[cache_key] = [] 
            return None
        
        print(f"Extracting evidence")
        prompt = f"""
        Extract distinct factual claims that help answer the question: '{sub_question}'.
        Only extract facts present in the text.
        
        CRITICAL INSTRUCTION: You MUST use the provided tool/function to output your response. 
        Do NOT output plain text or markdown. Call the tool with the 'claims' field.
        
        Text:
        {text}
        """
        await asyncio.to_thread(time.sleep, random.uniform(3, 7))
        extraction = await claim_llm.ainvoke(prompt)
        
        if extraction and extraction.claims:
            claims_str_list = [c.claim for c in extraction.claims]
            cache[cache_key] = claims_str_list
            return {
                "source_id": str(uuid.uuid4())[:8],
                "question": sub_question,
                "url": url,
                "claims": claims_str_list
            }
        else:
            cache[cache_key] = [] 
            return None
    except Exception as e:
        print(f"Failed on {url}: {e}")
        return None
