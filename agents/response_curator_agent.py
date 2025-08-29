"""
Response Curator Agent for generating comprehensive research reports.
Uses GPT-4o-mini to synthesize research findings into professional reports.
"""

import json
import logging
from typing import List, Dict, Any
from datetime import datetime

from openai import OpenAI
from pydantic import ValidationError

from config import OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE, INCLUDE_CITATIONS
from models import (
    ResearchQuery, ResearchPlan, CompetitorInfo, GapAnalysis, SearchResult,
    CompetitiveInsights, ResearchReport, AgentResponse, ResearchStatus
)

logger = logging.getLogger(__name__)

class ResponseCuratorAgent:
    """
    Agent responsible for curating and generating comprehensive research reports.
    
    Creates:
    - Executive summaries
    - Competitive insights and recommendations
    - Structured research reports
    - Professional markdown output
    - Strategic recommendations
    """
    
    def __init__(self):
        """Initialize the Response Curator Agent with GPT-4o-mini."""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for ResponseCuratorAgent")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.name = "ResponseCuratorAgent"
        
        logger.info(f"ResponseCuratorAgent initialized with model: {self.model}")
    
    def generate_competitive_insights(
        self,
        competitors: List[CompetitorInfo],
        plan: ResearchPlan,
        gap_analysis: GapAnalysis
    ) -> AgentResponse:
        """
        Generate strategic competitive insights and recommendations.
        
        Args:
            competitors: List of competitor information
            plan: Original research plan
            gap_analysis: Gap analysis results
            
        Returns:
            AgentResponse with CompetitiveInsights
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Generating competitive insights for {len(competitors)} competitors...")
            
            # Prepare competitor summary for analysis
            competitor_summary = self._create_competitor_summary(competitors)
            
            # Generate insights using GPT-4o-mini
            insights = self._generate_insights(competitor_summary, plan, gap_analysis)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Competitive insights generated in {execution_time:.2f}s")
            
            return AgentResponse(
                agent_name=self.name,
                status=ResearchStatus.COMPLETED,
                data={"competitive_insights": insights.model_dump()},
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error generating competitive insights: {e}")
            
            return AgentResponse(
                agent_name=self.name,
                status=ResearchStatus.FAILED,
                error=f"Insights generation failed: {str(e)}",
                execution_time_seconds=execution_time
            )
    
    def create_research_report(
        self,
        query: ResearchQuery,
        plan: ResearchPlan,
        competitors: List[CompetitorInfo],
        gap_analysis: GapAnalysis,
        insights: CompetitiveInsights,
        search_results: List[SearchResult],
        research_duration: float
    ) -> AgentResponse:
        """
        Create a comprehensive research report.
        
        Args:
            query: Original research query
            plan: Research plan executed
            competitors: Competitor information found
            gap_analysis: Gap analysis results  
            insights: Competitive insights generated
            search_results: Raw search results
            research_duration: Total research time in seconds
            
        Returns:
            AgentResponse with complete ResearchReport
        """
        start_time = datetime.now()
        
        try:
            logger.info("Creating comprehensive research report...")
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(
                query, competitors, insights, gap_analysis
            )
            
            # Create methodology description
            methodology = self._create_methodology_description(plan, len(search_results))
            
            # Identify data sources
            data_sources = self._extract_data_sources(search_results)
            
            # Generate limitations
            limitations = self._generate_limitations(gap_analysis)
            
            # Create next steps
            next_steps = self._generate_next_steps(gap_analysis, insights)
            
            # Create complete research report
            report = ResearchReport(
                query=query,
                plan=plan,
                competitors=competitors,
                gap_analysis=gap_analysis,
                insights=insights,
                executive_summary=executive_summary,
                methodology=methodology,
                data_sources=data_sources,
                limitations=limitations,
                next_steps=next_steps,
                total_searches_performed=len(set(r.query for r in search_results)),
                research_duration_seconds=research_duration
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Research report created in {execution_time:.2f}s")
            
            return AgentResponse(
                agent_name=self.name,
                status=ResearchStatus.COMPLETED,
                data={"research_report": report.model_dump()},
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error creating research report: {e}")
            
            return AgentResponse(
                agent_name=self.name,
                status=ResearchStatus.FAILED,
                error=f"Report creation failed: {str(e)}",
                execution_time_seconds=execution_time
            )
    
    def format_markdown_report(self, report: ResearchReport) -> str:
        """
        Format the research report as professional markdown.
        
        Args:
            report: Complete research report
            
        Returns:
            Formatted markdown string
        """
        try:
            timestamp = report.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC")
            duration_minutes = report.research_duration_seconds / 60 if report.research_duration_seconds else 0
            
            markdown = f"""# Competitive Research Report

**Generated:** {timestamp}  
**Research Duration:** {duration_minutes:.1f} minutes  
**Searches Performed:** {report.total_searches_performed}  
**Competitors Analyzed:** {len(report.competitors)}

---

## Executive Summary

{report.executive_summary}

## Research Objective

**Query:** {report.query.query}  
**Research Depth:** {report.query.depth}  
**Main Objective:** {report.plan.main_objective}

### Key Research Questions
{chr(10).join(f'- {q}' for q in report.plan.research_questions)}

---

## Competitive Landscape

### Competitors Identified ({len(report.competitors)})

"""
            
            # Add competitor profiles
            for i, comp in enumerate(report.competitors, 1):
                markdown += f"""
#### {i}. {comp.name}

- **Website:** {comp.website or 'Not available'}
- **Description:** {comp.description or 'Not available'}
- **Products:** {', '.join(comp.products) if comp.products else 'Not specified'}
- **Target Market:** {comp.target_market or 'Not specified'}
- **Market Position:** {comp.market_position or 'Not specified'}

"""
                
                if comp.key_features:
                    markdown += f"**Key Features:**\n{chr(10).join(f'- {feature}' for feature in comp.key_features[:5])}\n\n"
                
                if comp.pricing_info:
                    markdown += f"**Pricing Information:**\n"
                    for plan_name, price in comp.pricing_info.items():
                        markdown += f"- {plan_name}: {price}\n"
                    markdown += "\n"
            
            # Add strategic insights
            markdown += f"""---

## Strategic Analysis

### Market Opportunities
{chr(10).join(f'- {opp}' for opp in report.insights.market_opportunities)}

### Competitive Advantages
{chr(10).join(f'- {adv}' for adv in report.insights.competitive_advantages)}

### Threats and Risks  
{chr(10).join(f'- {threat}' for threat in report.insights.threats_and_risks)}

### Strategic Recommendations
{chr(10).join(f'- {rec}' for rec in report.insights.strategic_recommendations)}

"""
            
            # Add gap analysis
            if report.gap_analysis.priority_gaps:
                markdown += f"""---

## Research Gaps Analysis

**Data Quality Score:** {report.gap_analysis.data_quality_score:.1%} if report.gap_analysis.data_quality_score else 'Not calculated'

### Priority Gaps
{chr(10).join(f'- {gap}' for gap in report.gap_analysis.priority_gaps)}

### Missing Information
{chr(10).join(f'- {info}' for info in report.gap_analysis.missing_information)}

"""
            
            # Add confidence scores
            if report.gap_analysis.confidence_scores:
                markdown += "### Confidence Levels\n"
                for area, score in report.gap_analysis.confidence_scores.items():
                    markdown += f"- **{area.title()}:** {score:.1%}\n"
                markdown += "\n"
            
            # Add methodology and limitations
            markdown += f"""---

## Methodology

{report.methodology}

## Limitations

{chr(10).join(f'- {limitation}' for limitation in report.limitations)}

## Next Steps

{chr(10).join(f'- {step}' for step in report.next_steps)}

"""
            
            # Add data sources if citation is enabled
            if INCLUDE_CITATIONS and report.data_sources:
                markdown += f"""---

## Data Sources

{chr(10).join(f'- {source}' for source in report.data_sources[:10])}
"""
            
            markdown += f"""
---

*Report generated by Competitive Research Multi-Agent System*  
*Powered by GPT-4o-mini and Tavily AI*
"""
            
            return markdown
            
        except Exception as e:
            logger.error(f"Error formatting markdown report: {e}")
            return f"# Research Report\n\nError formatting report: {str(e)}\n\nRaw data:\n{json.dumps(report.model_dump(), indent=2, default=str)}"
    
    def _create_competitor_summary(self, competitors: List[CompetitorInfo]) -> str:
        """Create a concise summary of competitors for analysis."""
        if not competitors:
            return "No competitors identified in the research."
        
        summary_parts = []
        for comp in competitors[:10]:  # Limit for token management
            summary = f"**{comp.name}**"
            if comp.website:
                summary += f" ({comp.website})"
            if comp.description:
                summary += f" - {comp.description[:150]}{'...' if len(comp.description) > 150 else ''}"
            
            details = []
            if comp.products:
                details.append(f"Products: {', '.join(comp.products[:3])}")
            if comp.pricing_info:
                details.append(f"Pricing available")
            if comp.key_features:
                details.append(f"{len(comp.key_features)} key features")
            
            if details:
                summary += f" | {' | '.join(details)}"
            
            summary_parts.append(summary)
        
        return "\n".join(summary_parts)
    
    def _generate_insights(
        self,
        competitor_summary: str,
        plan: ResearchPlan,
        gap_analysis: GapAnalysis
    ) -> CompetitiveInsights:
        """Generate strategic insights using GPT-4o-mini."""
        
        prompt = f"""Analyze this competitive landscape and provide strategic insights:

RESEARCH OBJECTIVE: {plan.main_objective}

COMPETITORS FOUND:
{competitor_summary}

DATA QUALITY SCORE: {gap_analysis.data_quality_score:.1%} if gap_analysis.data_quality_score else 'Not calculated'

CONFIDENCE LEVELS:
{chr(10).join(f'- {area}: {score:.1%}' for area, score in gap_analysis.confidence_scores.items())}

Based on this competitive analysis, identify:

1. MARKET OPPORTUNITIES: Gaps or underserved areas in the market
2. COMPETITIVE ADVANTAGES: Potential advantages to leverage  
3. THREATS AND RISKS: Competitive threats to be aware of
4. STRATEGIC RECOMMENDATIONS: Actionable strategic advice
5. POSITIONING SUGGESTIONS: How to position in this market
6. FEATURE GAPS: Missing features or capabilities in the market
7. PRICING INSIGHTS: Pricing strategy observations

Focus on actionable insights that can inform business strategy and competitive positioning.

Format as JSON:
{{
    "market_opportunities": ["Opportunity 1", "Opportunity 2"],
    "competitive_advantages": ["Advantage 1", "Advantage 2"], 
    "threats_and_risks": ["Threat 1", "Threat 2"],
    "strategic_recommendations": ["Recommendation 1", "Recommendation 2"],
    "positioning_suggestions": ["Position 1", "Position 2"],
    "feature_gaps": ["Gap 1", "Gap 2"],
    "pricing_insights": ["Insight 1", "Insight 2"]
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strategic business analyst specializing in competitive intelligence. Provide actionable strategic insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"}
            )
            
            insights_data = json.loads(response.choices[0].message.content)
            
            return CompetitiveInsights(
                market_opportunities=insights_data.get("market_opportunities", []),
                competitive_advantages=insights_data.get("competitive_advantages", []),
                threats_and_risks=insights_data.get("threats_and_risks", []),
                strategic_recommendations=insights_data.get("strategic_recommendations", []),
                positioning_suggestions=insights_data.get("positioning_suggestions", []),
                feature_gaps=insights_data.get("feature_gaps", []),
                pricing_insights=insights_data.get("pricing_insights", [])
            )
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            
            # Fallback insights
            return CompetitiveInsights(
                market_opportunities=["Identify underserved market segments"],
                competitive_advantages=["Leverage unique capabilities"],
                threats_and_risks=["Monitor competitive actions"],
                strategic_recommendations=["Continue market research", "Develop differentiation strategy"],
                positioning_suggestions=["Focus on unique value proposition"],
                feature_gaps=["Analyze feature completeness"],
                pricing_insights=["Research competitive pricing"]
            )
    
    def _generate_executive_summary(
        self,
        query: ResearchQuery,
        competitors: List[CompetitorInfo],
        insights: CompetitiveInsights,
        gap_analysis: GapAnalysis
    ) -> str:
        """Generate executive summary using GPT-4o-mini."""
        
        competitor_names = [comp.name for comp in competitors[:5]]
        
        prompt = f"""Create a concise executive summary for this competitive research:

RESEARCH QUERY: {query.query}
COMPETITORS FOUND: {len(competitors)} ({', '.join(competitor_names)})
DATA QUALITY: {gap_analysis.data_quality_score:.1%} if gap_analysis.data_quality_score else 'Variable'

KEY INSIGHTS:
- Market Opportunities: {len(insights.market_opportunities)}
- Strategic Recommendations: {len(insights.strategic_recommendations)}
- Identified Threats: {len(insights.threats_and_risks)}

Write a 2-3 paragraph executive summary that covers:
1. What was researched and key findings
2. Main competitive landscape insights
3. Strategic implications and recommendations

Keep it concise but comprehensive, suitable for executive decision-making."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are writing an executive summary for competitive research. Be concise, strategic, and actionable."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            
            # Fallback summary
            return f"""This competitive research analyzed {len(competitors)} competitors in the {query.query} space. The research identified key market players, their positioning, and strategic opportunities. 

Based on the analysis, several market opportunities and competitive advantages were identified, along with potential threats and risks. The findings suggest specific strategic recommendations for competitive positioning and market entry.

Further research is recommended to address identified gaps and enhance competitive intelligence."""
    
    def _create_methodology_description(self, plan: ResearchPlan, total_results: int) -> str:
        """Create methodology description."""
        return f"""This competitive research employed a multi-agent approach using GPT-4o-mini and Tavily AI web search technology.

**Research Process:**
1. Strategic planning based on the research query
2. Systematic web search using {len(plan.search_keywords)} keywords across {len(plan.priority_areas)} priority areas
3. Automated data extraction and competitor profiling
4. Gap analysis to identify missing information
5. Synthesis of findings into strategic insights

**Data Collection:**
- {plan.estimated_searches} planned searches executed
- {total_results} search results analyzed
- Focus areas: {', '.join(plan.priority_areas)}
- Search depth: Advanced web search with AI-powered content extraction

**Analysis Methods:**
- Automated competitor profiling and feature extraction
- Gap analysis with confidence scoring
- Strategic insight generation using AI analysis
- Cross-validation of findings across multiple sources"""
    
    def _extract_data_sources(self, search_results: List[SearchResult]) -> List[str]:
        """Extract unique data sources from search results."""
        sources = set()
        
        for result in search_results:
            if result.url and result.url.startswith('http'):
                # Extract domain from URL
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(result.url).netloc
                    sources.add(domain)
                except:
                    sources.add(result.url[:50] + "..." if len(result.url) > 50 else result.url)
        
        # Add search method sources
        sources.add("Tavily AI Web Search")
        sources.add("GPT-4o-mini Analysis")
        
        return sorted(list(sources))
    
    def _generate_limitations(self, gap_analysis: GapAnalysis) -> List[str]:
        """Generate research limitations based on gap analysis."""
        limitations = []
        
        if gap_analysis.data_quality_score and gap_analysis.data_quality_score < 0.7:
            limitations.append(f"Data completeness score of {gap_analysis.data_quality_score:.1%} indicates some information gaps")
        
        if gap_analysis.missing_information:
            limitations.append(f"Missing critical information in {len(gap_analysis.missing_information)} areas")
        
        if gap_analysis.confidence_scores:
            low_confidence_areas = [area for area, score in gap_analysis.confidence_scores.items() if score < 0.6]
            if low_confidence_areas:
                limitations.append(f"Lower confidence in data for: {', '.join(low_confidence_areas)}")
        
        # Add standard limitations
        limitations.extend([
            "Information accuracy dependent on publicly available sources",
            "Market conditions and competitor data subject to rapid change",
            "Some proprietary information not accessible through public research"
        ])
        
        return limitations[:5]  # Limit to 5 key limitations
    
    def _generate_next_steps(self, gap_analysis: GapAnalysis, insights: CompetitiveInsights) -> List[str]:
        """Generate recommended next steps."""
        next_steps = []
        
        # Add steps from gap analysis
        if gap_analysis.suggested_queries:
            next_steps.append("Conduct additional research using suggested follow-up queries")
        
        if gap_analysis.priority_gaps:
            next_steps.append("Address priority information gaps for more complete analysis")
        
        # Add strategic next steps from insights
        if insights.strategic_recommendations:
            next_steps.append("Implement strategic recommendations based on competitive analysis")
        
        # Add standard next steps
        next_steps.extend([
            "Monitor competitor activities and market developments continuously",
            "Validate findings through direct market research or customer interviews",
            "Develop detailed competitive response strategies",
            "Schedule regular competitive intelligence updates"
        ])
        
        return next_steps[:6]  # Limit to 6 next steps