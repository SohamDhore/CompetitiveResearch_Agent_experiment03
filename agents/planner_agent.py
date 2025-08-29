"""
Planning Agent for creating strategic research plans.
Uses GPT-4o-mini to analyze queries and create comprehensive research strategies.
"""

import json
import logging
from typing import List, Dict, Any
from datetime import datetime

from openai import OpenAI
from pydantic import ValidationError

from config import OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE
from models import ResearchQuery, ResearchPlan, AgentResponse, ResearchStatus

logger = logging.getLogger(__name__)

class PlannerAgent:
    """
    Agent responsible for creating strategic research plans.
    
    Takes a research query and creates:
    - Main research objective
    - Key research questions
    - Priority areas to investigate
    - Search keywords and phrases
    - Specific competitor names to research
    """
    
    def __init__(self):
        """Initialize the Planning Agent with GPT-4o-mini."""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for PlannerAgent")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.name = "PlannerAgent"
        
        logger.info(f"PlannerAgent initialized with model: {self.model}")
    
    def create_research_plan(self, query: ResearchQuery) -> AgentResponse:
        """
        Create a comprehensive research plan based on the query.
        
        Args:
            query: ResearchQuery containing the research request
            
        Returns:
            AgentResponse with ResearchPlan or error information
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Creating research plan for: {query.query[:50]}...")
            prompt = self._build_planning_prompt(query)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a strategic research planner specializing in competitive analysis. Create detailed, actionable research plans."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            plan_data = json.loads(response.choices[0].message.content)
            logger.debug(f"Raw plan data: {plan_data}")
            
            # Create ResearchPlan object
            research_plan = self._parse_plan_response(plan_data, query)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Research plan created successfully in {execution_time:.2f}s")
            
            return AgentResponse(
                agent_name=self.name,
                status=ResearchStatus.COMPLETED,
                data={"research_plan": research_plan.model_dump()},
                execution_time_seconds=execution_time
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return self._create_fallback_plan(query, start_time, f"JSON parsing error: {str(e)}")
            
        except ValidationError as e:
            logger.error(f"Plan validation error: {e}")
            return self._create_fallback_plan(query, start_time, f"Validation error: {str(e)}")
            
        except Exception as e:
            logger.error(f"Unexpected error in create_research_plan: {e}")
            return self._create_fallback_plan(query, start_time, f"Unexpected error: {str(e)}")
    
    def _build_planning_prompt(self, query: ResearchQuery) -> str:
        """Build the prompt for research plan creation."""
        
        focus_areas_text = ""
        if query.focus_areas:
            focus_areas_text = f"\nSpecific focus areas requested: {', '.join(query.focus_areas)}"
        
        exclude_text = ""
        if query.exclude_competitors:
            exclude_text = f"\nExclude these competitors: {', '.join(query.exclude_competitors)}"
        
        depth_instructions = {
            "basic": "Focus on 3-5 main competitors and essential information only.",
            "standard": "Provide comprehensive analysis of 5-8 competitors with detailed information.",
            "comprehensive": "Conduct thorough research of 8-12 competitors with deep market analysis."
        }
        
        return f"""You are creating a strategic competitive research plan to find and analyze companies that operate in the same market space.

Research Query: "{query.query}"
Research Depth: {query.depth} - {depth_instructions.get(query.depth, "")}
Maximum Results per Search: {query.max_results}{focus_areas_text}{exclude_text}

IMPORTANT: You are looking for companies that provide products/services in the "{query.query}" market space, NOT companies that provide competitive research services.

Create a detailed research plan with:

1. MAIN OBJECTIVE: Clear, specific goal to identify and analyze companies in the "{query.query}" market
2. RESEARCH QUESTIONS: 5-8 key questions about competitors in this specific market
3. PRIORITY AREAS: Specific areas to investigate (e.g., pricing, features, market position, funding, technology, customer base, partnerships)
4. SEARCH KEYWORDS: 8-12 strategic keywords and phrases that identify companies in this market (avoid words like "competitors", "analysis", "research")
5. COMPETITOR NAMES: Specific company/product names in this market (if mentioned or inferable)

Focus on finding actual companies that operate in the "{query.query}" space, not companies that provide research services.

Format your response as JSON with this exact structure:
{{
    "main_objective": "Clear, specific research objective",
    "research_questions": [
        "Question 1 about competitor positioning",
        "Question 2 about market share",
        "Question 3 about pricing strategies",
        "Question 4 about product features",
        "Question 5 about target customers"
    ],
    "priority_areas": [
        "pricing",
        "features",
        "market_position",
        "technology",
        "funding"
    ],
    "search_keywords": [
        "primary keyword",
        "competitor analysis keyword",
        "market research phrase",
        "industry-specific term"
    ],
    "competitor_names": [
        "Specific Company 1",
        "Specific Company 2"
    ]
}}"""
    
    def _parse_plan_response(self, plan_data: Dict[str, Any], query: ResearchQuery) -> ResearchPlan:
        """Parse the GPT response into a ResearchPlan object."""
        
        # Calculate estimated searches based on plan complexity
        estimated_searches = (
            len(plan_data.get("priority_areas", [])) * 2 +
            len(plan_data.get("competitor_names", [])) +
            len(plan_data.get("search_keywords", [])) // 2
        )
        estimated_searches = max(5, min(estimated_searches, 25))  # Keep reasonable bounds
        
        return ResearchPlan(
            main_objective=plan_data.get("main_objective", query.query),
            research_questions=plan_data.get("research_questions", [f"Who are the main competitors for {query.query}?"]),
            priority_areas=plan_data.get("priority_areas", ["market_position", "features", "pricing"]),
            search_keywords=plan_data.get("search_keywords", [query.query]),
            competitor_names=plan_data.get("competitor_names", []),
            estimated_searches=estimated_searches
        )
    
    def _create_fallback_plan(self, query: ResearchQuery, start_time: datetime, error_msg: str) -> AgentResponse:
        """Create a basic fallback plan when the main process fails."""
        
        logger.warning(f"Creating fallback plan due to: {error_msg}")
        
        fallback_plan = ResearchPlan(
            main_objective=f"Competitive analysis for: {query.query}",
            research_questions=[
                f"Who are the main competitors in the {query.query} space?",
                "What are their key products and services?", 
                "How do they price their offerings?",
                "What are their main competitive advantages?",
                "Who is their target market?"
            ],
            priority_areas=query.focus_areas or ["market_position", "features", "pricing", "target_market"],
            search_keywords=[query.query] + (query.focus_areas or ["competitors", "market analysis"]),
            competitor_names=[],
            estimated_searches=8
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return AgentResponse(
            agent_name=self.name,
            status=ResearchStatus.COMPLETED,
            data={"research_plan": fallback_plan.model_dump()},
            error=f"Used fallback plan: {error_msg}",
            execution_time_seconds=execution_time
        )
    
    def refine_plan(self, plan: ResearchPlan, feedback: str) -> AgentResponse:
        """
        Refine an existing research plan based on feedback.
        
        Args:
            plan: Existing ResearchPlan to refine
            feedback: Feedback or additional requirements
            
        Returns:
            AgentResponse with refined ResearchPlan
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Refining research plan based on feedback: {feedback[:50]}...")
            
            prompt = f"""Refine this existing research plan based on new feedback:

CURRENT PLAN:
Main Objective: {plan.main_objective}
Research Questions: {json.dumps(plan.research_questions, indent=2)}
Priority Areas: {json.dumps(plan.priority_areas)}
Search Keywords: {json.dumps(plan.search_keywords)}
Competitor Names: {json.dumps(plan.competitor_names)}

FEEDBACK/REQUIREMENTS:
{feedback}

Create an improved plan that addresses the feedback while maintaining the core research objective. 

Format as JSON with the same structure as the original plan."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are refining a competitive research plan based on feedback. Improve the plan while maintaining its core objectives."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"}
            )
            
            refined_data = json.loads(response.choices[0].message.content)
            refined_plan = ResearchPlan(
                main_objective=refined_data.get("main_objective", plan.main_objective),
                research_questions=refined_data.get("research_questions", plan.research_questions),
                priority_areas=refined_data.get("priority_areas", plan.priority_areas),
                search_keywords=refined_data.get("search_keywords", plan.search_keywords),
                competitor_names=refined_data.get("competitor_names", plan.competitor_names),
                estimated_searches=len(refined_data.get("priority_areas", plan.priority_areas)) * 2 + len(refined_data.get("competitor_names", plan.competitor_names))
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Plan refined successfully in {execution_time:.2f}s")
            
            return AgentResponse(
                agent_name=self.name,
                status=ResearchStatus.COMPLETED,
                data={"research_plan": refined_plan.model_dump()},
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error refining plan: {e}")
            
            return AgentResponse(
                agent_name=self.name,
                status=ResearchStatus.FAILED,
                error=f"Failed to refine plan: {str(e)}",
                execution_time_seconds=execution_time
            )