"""
Gap Analysis Agent for identifying research gaps and missing information.
Uses GPT-4o-mini to analyze research completeness and suggest improvements.
"""

import json
import logging
from typing import List, Dict, Any
from datetime import datetime

from openai import OpenAI
from pydantic import ValidationError

from config import OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE
from models import (
    ResearchPlan, CompetitorInfo, SearchResult, GapAnalysis, 
    AgentResponse, ResearchStatus
)

logger = logging.getLogger(__name__)

class GapAnalyzerAgent:
    """
    Agent responsible for analyzing research gaps and completeness.
    
    Analyzes:
    - Missing critical information
    - Incomplete data areas  
    - Confidence levels by topic
    - Suggested follow-up queries
    - Priority gaps to address
    """
    
    def __init__(self):
        """Initialize the Gap Analysis Agent with GPT-4o-mini."""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for GapAnalyzerAgent")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.name = "GapAnalyzerAgent"
        
        logger.info(f"GapAnalyzerAgent initialized with model: {self.model}")
    
    def analyze_research_gaps(
        self, 
        plan: ResearchPlan,
        competitors: List[CompetitorInfo],
        search_results: List[SearchResult]
    ) -> AgentResponse:
        """
        Analyze gaps in the research data.
        
        Args:
            plan: Original research plan
            competitors: List of discovered competitors
            search_results: Raw search results from web searcher
            
        Returns:
            AgentResponse with GapAnalysis or error information
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Analyzing research gaps for {len(competitors)} competitors...")
            
            # Create comprehensive analysis of current data
            data_summary = self._create_data_summary(plan, competitors, search_results)
            
            # Analyze gaps using GPT-4o-mini
            gap_analysis = self._perform_gap_analysis(data_summary, plan)
            
            # Calculate overall data quality score
            quality_score = self._calculate_data_quality_score(competitors, plan)
            gap_analysis.data_quality_score = quality_score
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Gap analysis completed in {execution_time:.2f}s with quality score: {quality_score:.2f}")
            
            return AgentResponse(
                agent_name=self.name,
                status=ResearchStatus.COMPLETED,
                data={"gap_analysis": gap_analysis.model_dump()},
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error in gap analysis: {e}")
            
            return AgentResponse(
                agent_name=self.name,
                status=ResearchStatus.FAILED,
                error=f"Gap analysis failed: {str(e)}",
                execution_time_seconds=execution_time
            )
    
    def _create_data_summary(
        self, 
        plan: ResearchPlan,
        competitors: List[CompetitorInfo], 
        search_results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Create a comprehensive summary of available data."""
        
        # Analyze competitor data completeness
        competitor_analysis = []
        for comp in competitors:
            analysis = {
                "name": comp.name,
                "has_website": bool(comp.website),
                "has_description": bool(comp.description),
                "products_count": len(comp.products),
                "has_pricing": bool(comp.pricing_info),
                "features_count": len(comp.key_features),
                "has_target_market": bool(comp.target_market),
                "has_market_position": bool(comp.market_position),
                "has_funding_info": bool(comp.funding_info),
                "recent_news_count": len(comp.recent_news),
                "completeness_score": self._calculate_competitor_completeness(comp)
            }
            competitor_analysis.append(analysis)
        
        # Analyze search coverage
        search_coverage = {
            "total_searches": len(set(result.query for result in search_results)),
            "total_results": len(search_results),
            "unique_sources": len(set(result.url for result in search_results if result.url)),
            "coverage_by_priority_area": {}
        }
        
        # Check coverage of priority areas
        for area in plan.priority_areas:
            area_results = [r for r in search_results if area.lower() in r.query.lower() or area.lower() in r.content.lower()]
            search_coverage["coverage_by_priority_area"][area] = len(area_results)
        
        return {
            "research_objective": plan.main_objective,
            "research_questions": plan.research_questions,
            "priority_areas": plan.priority_areas,
            "planned_competitors": plan.competitor_names,
            "found_competitors": len(competitors),
            "competitor_analysis": competitor_analysis,
            "search_coverage": search_coverage,
            "estimated_vs_actual_searches": {
                "estimated": plan.estimated_searches,
                "actual": len(set(result.query for result in search_results))
            }
        }
    
    def _calculate_competitor_completeness(self, competitor: CompetitorInfo) -> float:
        """Calculate completeness score for a single competitor (0-1)."""
        fields_to_check = [
            competitor.website,
            competitor.description, 
            competitor.products,
            competitor.pricing_info,
            competitor.key_features,
            competitor.target_market,
            competitor.market_position
        ]
        
        filled_fields = sum(1 for field in fields_to_check if field)
        return filled_fields / len(fields_to_check)
    
    def _perform_gap_analysis(self, data_summary: Dict[str, Any], plan: ResearchPlan) -> GapAnalysis:
        """Perform detailed gap analysis using GPT-4o-mini."""
        
        prompt = f"""Analyze the completeness and gaps in this competitive research data:

RESEARCH OBJECTIVE: {data_summary['research_objective']}

RESEARCH QUESTIONS TO ANSWER:
{chr(10).join(f'- {q}' for q in data_summary['research_questions'])}

PRIORITY AREAS: {', '.join(data_summary['priority_areas'])}

RESEARCH FINDINGS:
- Competitors found: {data_summary['found_competitors']} (planned: {len(data_summary['planned_competitors'])})
- Total search results: {data_summary['search_coverage']['total_results']}
- Unique sources: {data_summary['search_coverage']['unique_sources']}

COMPETITOR COMPLETENESS:
{chr(10).join(f"- {comp['name']}: {comp['completeness_score']:.1%} complete" for comp in data_summary['competitor_analysis'])}

PRIORITY AREA COVERAGE:
{chr(10).join(f"- {area}: {count} results" for area, count in data_summary['search_coverage']['coverage_by_priority_area'].items())}

Analyze this research for:

1. MISSING CRITICAL INFORMATION: What essential information is completely missing?
2. INCOMPLETE AREAS: Which areas have some data but need more detail?
3. CONFIDENCE SCORES: Rate confidence level (0-1) for each priority area based on data quality and completeness
4. SUGGESTED QUERIES: Specific follow-up searches needed to fill gaps
5. PRIORITY GAPS: Most important gaps to address first

Consider which research questions remain unanswered and what information would be most valuable for competitive positioning.

Format as JSON:
{{
    "missing_information": ["Critical info 1", "Critical info 2"],
    "incomplete_areas": {{
        "pricing": ["specific gaps in pricing data"],
        "features": ["specific gaps in feature data"]
    }},
    "confidence_scores": {{
        "pricing": 0.7,
        "features": 0.8,
        "market_position": 0.6
    }},
    "suggested_queries": ["Specific search query 1", "Specific search query 2"],
    "priority_gaps": ["Highest priority gap 1", "Highest priority gap 2"]
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research gap analysis expert. Identify missing information and suggest specific improvements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"}
            )
            
            gap_data = json.loads(response.choices[0].message.content)
            
            return GapAnalysis(
                missing_information=gap_data.get("missing_information", []),
                incomplete_areas=gap_data.get("incomplete_areas", {}),
                confidence_scores=gap_data.get("confidence_scores", {}),
                suggested_queries=gap_data.get("suggested_queries", []),
                priority_gaps=gap_data.get("priority_gaps", [])
            )
            
        except Exception as e:
            logger.error(f"Error in GPT gap analysis: {e}")
            
            # Create fallback gap analysis
            return self._create_fallback_gap_analysis(data_summary, plan)
    
    def _create_fallback_gap_analysis(self, data_summary: Dict[str, Any], plan: ResearchPlan) -> GapAnalysis:
        """Create basic gap analysis when GPT analysis fails."""
        
        missing_info = []
        incomplete_areas = {}
        confidence_scores = {}
        suggested_queries = []
        priority_gaps = []
        
        # Basic analysis of missing information
        if data_summary['found_competitors'] == 0:
            missing_info.append("No competitors identified")
            priority_gaps.append("Identify main competitors in the market")
        
        if data_summary['found_competitors'] < 3:
            missing_info.append("Insufficient competitor coverage")
            suggested_queries.append(f"{plan.search_keywords[0] if plan.search_keywords else 'competitors'} market leaders")
        
        # Analyze incomplete areas based on low coverage
        for area, count in data_summary['search_coverage']['coverage_by_priority_area'].items():
            if count == 0:
                incomplete_areas[area] = [f"No data found for {area}"]
                confidence_scores[area] = 0.0
                suggested_queries.append(f"{area} analysis {plan.search_keywords[0] if plan.search_keywords else ''}")
            elif count < 3:
                incomplete_areas[area] = [f"Limited data for {area}"]
                confidence_scores[area] = 0.3
            else:
                confidence_scores[area] = 0.7
        
        # Add general priority gaps
        if not priority_gaps:
            priority_gaps = [
                "Expand competitor identification",
                "Gather more detailed pricing information",
                "Collect feature comparison data"
            ]
        
        return GapAnalysis(
            missing_information=missing_info,
            incomplete_areas=incomplete_areas,
            confidence_scores=confidence_scores,
            suggested_queries=suggested_queries[:8],  # Limit to 8 queries
            priority_gaps=priority_gaps[:5]  # Limit to 5 priority gaps
        )
    
    def _calculate_data_quality_score(self, competitors: List[CompetitorInfo], plan: ResearchPlan) -> float:
        """Calculate overall data quality score (0-1)."""
        if not competitors:
            return 0.0
        
        # Factor 1: Competitor completeness (40%)
        completeness_scores = [self._calculate_competitor_completeness(comp) for comp in competitors]
        avg_completeness = sum(completeness_scores) / len(completeness_scores)
        completeness_factor = avg_completeness * 0.4
        
        # Factor 2: Coverage breadth (30%)
        expected_competitors = max(3, len(plan.competitor_names))
        coverage_ratio = min(len(competitors) / expected_competitors, 1.0)
        coverage_factor = coverage_ratio * 0.3
        
        # Factor 3: Information depth (30%)
        depth_indicators = []
        for comp in competitors:
            indicators = [
                bool(comp.pricing_info),
                len(comp.key_features) > 2,
                bool(comp.target_market),
                bool(comp.market_position),
                len(comp.products) > 1
            ]
            depth_indicators.append(sum(indicators) / len(indicators))
        
        avg_depth = sum(depth_indicators) / len(depth_indicators) if depth_indicators else 0
        depth_factor = avg_depth * 0.3
        
        total_score = completeness_factor + coverage_factor + depth_factor
        return round(total_score, 2)
    
    def generate_improvement_recommendations(
        self,
        gap_analysis: GapAnalysis,
        competitors: List[CompetitorInfo]
    ) -> AgentResponse:
        """
        Generate specific recommendations for improving the research.
        
        Args:
            gap_analysis: Results from gap analysis
            competitors: Current competitor data
            
        Returns:
            AgentResponse with improvement recommendations
        """
        start_time = datetime.now()
        
        try:
            logger.info("Generating improvement recommendations...")
            
            competitor_names = [comp.name for comp in competitors]
            
            prompt = f"""Based on this gap analysis, provide specific improvement recommendations:

CURRENT COMPETITORS: {', '.join(competitor_names[:5])}

PRIORITY GAPS:
{chr(10).join(f'- {gap}' for gap in gap_analysis.priority_gaps)}

CONFIDENCE SCORES:
{chr(10).join(f'- {area}: {score:.1%}' for area, score in gap_analysis.confidence_scores.items())}

SUGGESTED QUERIES:
{chr(10).join(f'- {query}' for query in gap_analysis.suggested_queries)}

Provide 5-7 actionable recommendations for:
1. Improving data collection strategy
2. Filling critical information gaps
3. Enhancing competitive intelligence
4. Next steps for market analysis
5. Research methodology improvements

Format as JSON array of strings:
["Recommendation 1", "Recommendation 2", ...]"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strategic research advisor. Provide actionable recommendations for improving competitive research."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE
            )
            
            # Parse recommendations
            content = response.choices[0].message.content
            
            # Try to extract JSON array
            import re
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                recommendations = json.loads(json_match.group())
            else:
                # Fallback: split by lines and clean
                recommendations = [
                    line.strip('- ').strip() 
                    for line in content.split('\n') 
                    if line.strip() and not line.strip().startswith('#')
                ][:7]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Generated {len(recommendations)} recommendations in {execution_time:.2f}s")
            
            return AgentResponse(
                agent_name=self.name,
                status=ResearchStatus.COMPLETED,
                data={"recommendations": recommendations},
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error generating recommendations: {e}")
            
            # Fallback recommendations
            fallback_recommendations = [
                "Conduct deeper searches for top 3 competitors",
                "Focus on collecting pricing information",
                "Gather more detailed feature comparisons",
                "Research recent company news and developments",
                "Analyze customer reviews and feedback",
                "Investigate partnership and acquisition activity",
                "Monitor competitor social media and marketing"
            ]
            
            return AgentResponse(
                agent_name=self.name,
                status=ResearchStatus.COMPLETED,
                data={"recommendations": fallback_recommendations},
                error=f"Used fallback recommendations: {str(e)}",
                execution_time_seconds=execution_time
            )