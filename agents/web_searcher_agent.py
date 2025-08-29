"""
Web Searcher Agent using Tavily AI for intelligent web search.
Performs comprehensive competitive research using Tavily's specialized search API.
"""

import json
import logging
import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from openai import OpenAI
from pydantic import ValidationError

from config import (
    OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE, TAVILY_API_KEY,
    MAX_SEARCH_RESULTS, TAVILY_SEARCH_DEPTH, TAVILY_TOPIC,
    TAVILY_INCLUDE_ANSWER, TAVILY_INCLUDE_IMAGES, REQUEST_TIMEOUT,
    MAX_RETRIES, MAX_CONCURRENT_SEARCHES
)
from models import (
    ResearchPlan, SearchResult, SearchResponse, CompetitorInfo, 
    AgentResponse, ResearchStatus
)

logger = logging.getLogger(__name__)

class WebSearcherAgent:
    """
    Agent responsible for web searching using Tavily AI.
    
    Performs comprehensive competitive research by:
    - Executing strategic searches based on research plans
    - Using Tavily AI's specialized web search capabilities
    - Extracting structured competitor information
    - Providing fallback capabilities when needed
    """
    
    def __init__(self):
        """Initialize the Web Searcher Agent with Tavily AI and GPT-4o-mini."""
        if not TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY is required for WebSearcherAgent")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for WebSearcherAgent")
        
        self.tavily_api_key = TAVILY_API_KEY
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.name = "WebSearcherAgent"
        
        # Tavily API configuration
        self.tavily_base_url = "https://api.tavily.com"
        self.search_endpoint = f"{self.tavily_base_url}/search"
        
        logger.info(f"WebSearcherAgent initialized with Tavily AI and model: {self.model}")
    
    async def execute_research(self, plan: ResearchPlan) -> AgentResponse:
        """
        Execute comprehensive web research based on the research plan.
        
        Args:
            plan: ResearchPlan containing search strategy
            
        Returns:
            AgentResponse with search results and extracted competitor info
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting web research execution for: {plan.main_objective[:50]}...")
            
            # Validate Tavily API key first
            if not await self._validate_tavily_api():
                return self._create_fallback_response(plan, start_time, "Tavily API validation failed")
            
            # Generate search queries from the plan
            search_queries = self._generate_search_queries(plan)
            logger.info(f"Generated {len(search_queries)} search queries")
            
            # Execute searches concurrently
            all_search_results = await self._execute_concurrent_searches(search_queries)
            
            # Extract competitor information from search results
            competitors = await self._extract_competitor_info(all_search_results, plan)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Research execution completed in {execution_time:.2f}s with {len(competitors)} competitors found")
            
            return AgentResponse(
                agent_name=self.name,
                status=ResearchStatus.COMPLETED,
                data={
                    "search_results": [result.model_dump() for result in all_search_results],
                    "competitors": [comp.model_dump() for comp in competitors],
                    "total_searches": len(search_queries),
                    "total_results": len(all_search_results)
                },
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error in research execution: {e}")
            return self._create_fallback_response(plan, start_time, f"Execution error: {str(e)}")
    
    async def _validate_tavily_api(self) -> bool:
        """Validate Tavily API key with a simple test request."""
        try:
            test_payload = {
                "api_key": self.tavily_api_key,
                "query": "test query",
                "max_results": 1
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(self.search_endpoint, json=test_payload) as response:
                    if response.status == 200:
                        logger.info("✅ Tavily API key validated successfully")
                        return True
                    elif response.status == 401:
                        logger.error("❌ Invalid Tavily API key")
                        return False
                    else:
                        logger.warning(f"Tavily API test returned status: {response.status}")
                        return False
                        
        except Exception as e:
            logger.warning(f"Could not validate Tavily API key: {e}")
            return False
    
    def _generate_search_queries(self, plan: ResearchPlan) -> List[str]:
        """Generate strategic search queries from the research plan."""
        queries = []
        
        # Priority area searches with main keywords
        for area in plan.priority_areas[:4]:  # Limit to top 4 areas
            for keyword in plan.search_keywords[:3]:  # Limit to top 3 keywords
                query = f"{keyword} {area} companies"
                queries.append(query)
        
        # Direct keyword searches for companies
        for keyword in plan.search_keywords[:3]:
            query = f"{keyword} companies list"
            queries.append(query)
        
        # Specific competitor searches
        for competitor in plan.competitor_names[:5]:  # Limit to 5 competitors
            query = f"{competitor} company profile products pricing features"
            queries.append(query)
        
        # Industry overview searches
        if plan.search_keywords:
            industry_query = f"{plan.search_keywords[0]} market leaders companies"
            queries.append(industry_query)
        
        # Remove duplicates and limit total queries
        unique_queries = list(dict.fromkeys(queries))
        limited_queries = unique_queries[:MAX_CONCURRENT_SEARCHES]
        
        logger.debug(f"Generated queries: {limited_queries}")
        return limited_queries
    
    async def _execute_concurrent_searches(self, queries: List[str]) -> List[SearchResult]:
        """Execute multiple searches concurrently using Tavily AI."""
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)
        
        async def search_with_semaphore(query: str) -> List[SearchResult]:
            async with semaphore:
                return await self._search_tavily(query)
        
        # Execute all searches concurrently
        search_tasks = [search_with_semaphore(query) for query in queries]
        results_lists = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions
        all_results = []
        for i, result in enumerate(results_lists):
            if isinstance(result, Exception):
                logger.error(f"Search failed for query '{queries[i]}': {result}")
                continue
            if isinstance(result, list):
                all_results.extend(result)
        
        logger.info(f"Completed {len(queries)} searches with {len(all_results)} total results")
        return all_results
    
    async def _search_tavily(self, query: str) -> List[SearchResult]:
        """Execute a single search using Tavily AI."""
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": TAVILY_SEARCH_DEPTH,
            "topic": TAVILY_TOPIC,
            "max_results": MAX_SEARCH_RESULTS,
            "include_answer": TAVILY_INCLUDE_ANSWER,
            "include_images": TAVILY_INCLUDE_IMAGES
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                ) as session:
                    async with session.post(self.search_endpoint, json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            return self._parse_tavily_response(query, data)
                        else:
                            error_text = await response.text()
                            logger.warning(f"Tavily API error {response.status} for query '{query}': {error_text}")
                            
                            if response.status == 429:  # Rate limit
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            else:
                                break
                                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1} for query: {query}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error searching Tavily for '{query}': {e}")
                break
        
        # Fallback to GPT-based search if Tavily fails
        logger.warning(f"Tavily search failed for '{query}', using GPT fallback")
        return await self._fallback_search(query)
    
    def _parse_tavily_response(self, query: str, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse Tavily API response into SearchResult objects."""
        try:
            results = []
            
            for item in data.get("results", []):
                result = SearchResult(
                    query=query,
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", "")[:500],  # Limit snippet length
                    content=item.get("content", ""),
                    score=item.get("score"),
                    published_date=item.get("published_date"),
                    source_type="web"
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error parsing Tavily response for query '{query}': {e}")
            return []
    
    async def _fallback_search(self, query: str) -> List[SearchResult]:
        """Fallback search using GPT-4o-mini knowledge when Tavily is unavailable."""
        try:
            prompt = f"""Based on your knowledge, provide information about: "{query}"

Focus on providing factual, current information about:
- Company websites and official sources
- Recent product updates and announcements  
- Pricing information when available
- Key features and capabilities
- Market positioning

Format as a JSON array with entries like:
[
    {{
        "title": "Company Name - Brief Description",
        "url": "https://company-website.com (if known)",
        "snippet": "Brief description of 100-200 words",
        "content": "More detailed information about the company, products, pricing, etc."
    }}
]

Provide 3-5 relevant entries if information is available."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a business information provider. Provide factual, structured information about companies and markets."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE
            )
            
            # Parse GPT response
            content = response.choices[0].message.content
            
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                results = []
                for item in data:
                    result = SearchResult(
                        query=query,
                        title=item.get("title", "GPT Knowledge Result"),
                        url=item.get("url", ""),
                        snippet=item.get("snippet", "")[:500],
                        content=item.get("content", ""),
                        source_type="knowledge_base"
                    )
                    results.append(result)
                
                return results
            
        except Exception as e:
            logger.error(f"Fallback search error for '{query}': {e}")
        
        return []
    
    async def _extract_competitor_info(self, search_results: List[SearchResult], plan: ResearchPlan) -> List[CompetitorInfo]:
        """Extract structured competitor information using GPT-4o-mini."""
        try:
            # Combine search results into a manageable text corpus
            content_blocks = []
            for result in search_results[:15]:  # Limit for token management
                content_block = f"Title: {result.title}\nURL: {result.url}\nContent: {result.snippet}\n"
                content_blocks.append(content_block)
            
            combined_content = "\n---\n".join(content_blocks)
            
            prompt = f"""Extract competitor information from these search results for: {plan.main_objective}

SEARCH RESULTS:
{combined_content}

Extract information for each competitor company found. For each competitor, identify:
- Company name
- Official website URL
- Company description  
- Products and services offered
- Pricing information (if mentioned)
- Key features and capabilities
- Target market or customer base
- Market positioning
- Recent news or developments
- Funding information (if available)

Format as a JSON array:
[
    {{
        "name": "Company Name",
        "website": "https://company.com",
        "description": "Company description...",
        "products": ["Product 1", "Product 2"],
        "pricing_info": {{"plan_name": "price_info"}},
        "key_features": ["Feature 1", "Feature 2"],
        "target_market": "Description of target market",
        "market_position": "Market positioning description",
        "recent_news": ["Recent development 1"],
        "funding_info": {{"stage": "Series A", "amount": "$10M"}},
        "employee_count": "50-100",
        "founded_year": 2020
    }}
]

Only include companies that are actual competitors or relevant to the research objective. Provide accurate information based on the search results."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data extraction specialist. Extract accurate, structured competitor information from search results."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Handle different response formats
            if isinstance(data, dict) and "competitors" in data:
                competitors_data = data["competitors"]
            elif isinstance(data, list):
                competitors_data = data
            else:
                # Try to find a list in the dict values
                competitors_data = []
                for value in data.values():
                    if isinstance(value, list):
                        competitors_data = value
                        break
            
            # Create CompetitorInfo objects
            competitors = []
            for comp_data in competitors_data[:10]:  # Limit to 10 competitors
                try:
                    competitor = CompetitorInfo(
                        name=comp_data.get("name", "Unknown Company"),
                        website=comp_data.get("website"),
                        description=comp_data.get("description"),
                        products=comp_data.get("products", []),
                        pricing_info=comp_data.get("pricing_info", {}),
                        key_features=comp_data.get("key_features", []),
                        target_market=comp_data.get("target_market"),
                        market_position=comp_data.get("market_position"),
                        recent_news=comp_data.get("recent_news", []),
                        funding_info=comp_data.get("funding_info"),
                        employee_count=comp_data.get("employee_count"),
                        founded_year=comp_data.get("founded_year")
                    )
                    competitors.append(competitor)
                except ValidationError as e:
                    logger.warning(f"Validation error for competitor data: {e}")
                    continue
            
            logger.info(f"Extracted information for {len(competitors)} competitors")
            return competitors
            
        except Exception as e:
            logger.error(f"Error extracting competitor info: {e}")
            return []
    
    def _create_fallback_response(self, plan: ResearchPlan, start_time: datetime, error_msg: str) -> AgentResponse:
        """Create a fallback response when search fails."""
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return AgentResponse(
            agent_name=self.name,
            status=ResearchStatus.FAILED,
            data={
                "search_results": [],
                "competitors": [],
                "total_searches": 0,
                "total_results": 0
            },
            error=error_msg,
            execution_time_seconds=execution_time
        )