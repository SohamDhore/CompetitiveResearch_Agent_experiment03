"""
Pydantic data models for the competitive research multi-agent system.
Provides type safety and data validation for all agent communications.
"""

from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

class ResearchDepth(str, Enum):
    """Research depth levels."""
    BASIC = "basic"
    STANDARD = "standard" 
    COMPREHENSIVE = "comprehensive"

class SearchDepth(str, Enum):
    """Tavily search depth options."""
    BASIC = "basic"
    ADVANCED = "advanced"

class ResearchStatus(str, Enum):
    """Research workflow status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

# Core Data Models

class ResearchQuery(BaseModel):
    """Input query for competitive research."""
    query: str = Field(..., description="The research query or question")
    depth: ResearchDepth = Field(default=ResearchDepth.STANDARD, description="Research depth level")
    focus_areas: Optional[List[str]] = Field(default=None, description="Specific areas to focus on")
    exclude_competitors: Optional[List[str]] = Field(default=None, description="Competitors to exclude")
    max_results: Optional[int] = Field(default=10, description="Maximum results per search")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError('Query must be at least 3 characters long')
        return v.strip()

class ResearchPlan(BaseModel):
    """Strategic research plan created by the Planning Agent."""
    main_objective: str = Field(..., description="Primary research objective")
    research_questions: List[str] = Field(..., description="Key questions to answer")
    priority_areas: List[str] = Field(..., description="Priority areas to investigate")
    search_keywords: List[str] = Field(..., description="Keywords for web searches")
    competitor_names: List[str] = Field(default_factory=list, description="Specific competitors to research")
    estimated_searches: int = Field(default=5, description="Estimated number of searches needed")
    created_at: datetime = Field(default_factory=datetime.now)

class SearchResult(BaseModel):
    """Individual search result from Tavily AI."""
    query: str = Field(..., description="Original search query")
    title: str = Field(..., description="Result title")
    url: str = Field(..., description="Source URL")
    snippet: str = Field(..., description="Brief excerpt or summary")
    content: str = Field(..., description="Full content or detailed information")
    score: Optional[float] = Field(default=None, description="Relevance score")
    published_date: Optional[str] = Field(default=None, description="Publication date if available")
    source_type: Optional[str] = Field(default=None, description="Type of source")

class SearchResponse(BaseModel):
    """Complete response from Tavily AI search."""
    query: str = Field(..., description="Original query")
    results: List[SearchResult] = Field(..., description="Search results")
    answer: Optional[str] = Field(default=None, description="AI-generated answer summary")
    images: List[str] = Field(default_factory=list, description="Related images")
    follow_up_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    response_time: Optional[float] = Field(default=None, description="Response time in seconds")
    total_results: int = Field(..., description="Total number of results")

class CompetitorInfo(BaseModel):
    """Structured information about a competitor."""
    name: str = Field(..., description="Company or product name")
    website: Optional[str] = Field(default=None, description="Official website URL")
    description: Optional[str] = Field(default=None, description="Company description")
    products: List[str] = Field(default_factory=list, description="Products and services")
    pricing_info: Dict[str, Any] = Field(default_factory=dict, description="Pricing information")
    key_features: List[str] = Field(default_factory=list, description="Key features and capabilities")
    target_market: Optional[str] = Field(default=None, description="Target market or audience")
    market_position: Optional[str] = Field(default=None, description="Market positioning")
    strengths: List[str] = Field(default_factory=list, description="Key strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Potential weaknesses")
    recent_news: List[str] = Field(default_factory=list, description="Recent news or updates")
    funding_info: Optional[Dict[str, Any]] = Field(default=None, description="Funding information")
    employee_count: Optional[str] = Field(default=None, description="Employee count range")
    founded_year: Optional[Union[int, str]] = Field(default=None, description="Year founded")

class GapAnalysis(BaseModel):
    """Analysis of research gaps and missing information."""
    missing_information: List[str] = Field(..., description="Critical missing information")
    incomplete_areas: Dict[str, List[str]] = Field(..., description="Areas with incomplete data")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores by area (0-1)")
    suggested_queries: List[str] = Field(..., description="Suggested follow-up queries")
    priority_gaps: List[str] = Field(..., description="Highest priority gaps to fill")
    data_quality_score: Optional[float] = Field(default=None, description="Overall data quality (0-1)")

class CompetitiveInsights(BaseModel):
    """Strategic insights and recommendations."""
    market_opportunities: List[str] = Field(..., description="Identified market opportunities")
    competitive_advantages: List[str] = Field(..., description="Potential competitive advantages")
    threats_and_risks: List[str] = Field(..., description="Competitive threats and risks")
    strategic_recommendations: List[str] = Field(..., description="Strategic recommendations")
    positioning_suggestions: List[str] = Field(..., description="Market positioning suggestions")
    feature_gaps: List[str] = Field(default_factory=list, description="Feature gaps in the market")
    pricing_insights: List[str] = Field(default_factory=list, description="Pricing strategy insights")

class ResearchReport(BaseModel):
    """Complete research report with all findings."""
    query: ResearchQuery = Field(..., description="Original research query")
    plan: ResearchPlan = Field(..., description="Research plan executed")
    competitors: List[CompetitorInfo] = Field(..., description="Competitor information")
    gap_analysis: GapAnalysis = Field(..., description="Gap analysis results")
    insights: CompetitiveInsights = Field(..., description="Strategic insights")
    executive_summary: str = Field(..., description="Executive summary")
    methodology: str = Field(..., description="Research methodology used")
    data_sources: List[str] = Field(..., description="Data sources consulted")
    limitations: List[str] = Field(..., description="Research limitations")
    next_steps: List[str] = Field(..., description="Recommended next steps")
    generated_at: datetime = Field(default_factory=datetime.now)
    total_searches_performed: int = Field(default=0, description="Total searches performed")
    research_duration_seconds: Optional[float] = Field(default=None, description="Research duration")

# Agent Communication Models

class AgentMessage(BaseModel):
    """Message passed between agents."""
    sender: str = Field(..., description="Sending agent name")
    recipient: str = Field(..., description="Receiving agent name")
    message_type: str = Field(..., description="Type of message")
    payload: Dict[str, Any] = Field(..., description="Message payload")
    timestamp: datetime = Field(default_factory=datetime.now)

class AgentResponse(BaseModel):
    """Response from an agent."""
    agent_name: str = Field(..., description="Agent name")
    status: ResearchStatus = Field(..., description="Response status")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time_seconds: Optional[float] = Field(default=None, description="Execution time")
    timestamp: datetime = Field(default_factory=datetime.now)

# Workflow Models

class WorkflowStep(BaseModel):
    """Individual step in the research workflow."""
    step_name: str = Field(..., description="Name of the workflow step")
    agent_name: str = Field(..., description="Agent responsible for this step")
    status: ResearchStatus = Field(default=ResearchStatus.PENDING)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    output_data: Optional[Dict[str, Any]] = Field(default=None)

class WorkflowExecution(BaseModel):
    """Complete workflow execution tracking."""
    workflow_id: str = Field(..., description="Unique workflow identifier")
    query: ResearchQuery = Field(..., description="Original research query")
    steps: List[WorkflowStep] = Field(..., description="Workflow steps")
    current_step: int = Field(default=0, description="Current step index")
    status: ResearchStatus = Field(default=ResearchStatus.PENDING)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = Field(default=None)
    total_duration_seconds: Optional[float] = Field(default=None)
    final_report: Optional[ResearchReport] = Field(default=None)

# Utility Models

class ConfigSummary(BaseModel):
    """Configuration summary for validation."""
    model: str = Field(..., description="LLM model being used")
    api_keys_configured: Dict[str, bool] = Field(..., description="API key configuration status")
    search_settings: Dict[str, Any] = Field(..., description="Search configuration")
    output_settings: Dict[str, Any] = Field(..., description="Output configuration")

class SystemMetrics(BaseModel):
    """System performance metrics."""
    total_queries_processed: int = Field(default=0)
    average_processing_time_seconds: Optional[float] = Field(default=None)
    success_rate: Optional[float] = Field(default=None)
    api_calls_made: Dict[str, int] = Field(default_factory=dict)
    errors_encountered: Dict[str, int] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)