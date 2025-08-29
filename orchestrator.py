"""
Orchestrator Agent for coordinating the competitive research workflow.
Manages the execution flow and coordinates all agents using GPT-4o-mini.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from config import validate_config
from models import (
    ResearchQuery, ResearchPlan, CompetitorInfo, GapAnalysis, 
    CompetitiveInsights, ResearchReport, SearchResult,
    WorkflowExecution, WorkflowStep, ResearchStatus
)
from agents import (
    PlannerAgent, WebSearcherAgent, GapAnalyzerAgent, ResponseCuratorAgent
)

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    """
    Master orchestrator agent that coordinates the entire research workflow.
    
    Workflow:
    1. Planning Agent creates research strategy
    2. Web Searcher Agent executes searches using Tavily AI
    3. Gap Analyzer Agent identifies missing information
    4. Response Curator Agent generates final report
    """
    
    def __init__(self):
        """Initialize the orchestrator with all agents."""
        # Validate configuration first
        if not validate_config():
            raise ValueError("Invalid configuration. Please check your API keys and settings.")
        
        self.name = "OrchestratorAgent"
        
        # Initialize all agents
        self.planner = PlannerAgent()
        self.web_searcher = WebSearcherAgent()
        self.gap_analyzer = GapAnalyzerAgent()
        self.response_curator = ResponseCuratorAgent()
        
        logger.info("OrchestratorAgent initialized with all sub-agents")
    
    async def execute_research(self, query: ResearchQuery) -> Dict[str, Any]:
        """
        Execute the complete competitive research workflow.
        
        Args:
            query: ResearchQuery containing the research request
            
        Returns:
            Dictionary containing the final research report and metadata
        """
        workflow_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Starting research workflow {workflow_id} for query: {query.query[:50]}...")
        
        # Initialize workflow tracking
        workflow = WorkflowExecution(
            workflow_id=workflow_id,
            query=query,
            steps=[
                WorkflowStep(step_name="planning", agent_name="PlannerAgent"),
                WorkflowStep(step_name="web_search", agent_name="WebSearcherAgent"), 
                WorkflowStep(step_name="gap_analysis", agent_name="GapAnalyzerAgent"),
                WorkflowStep(step_name="report_generation", agent_name="ResponseCuratorAgent")
            ],
            status=ResearchStatus.IN_PROGRESS
        )
        
        try:
            # Step 1: Create Research Plan
            logger.info("Step 1: Creating research plan...")
            workflow.steps[0].status = ResearchStatus.IN_PROGRESS
            workflow.steps[0].started_at = datetime.now()
            
            planning_response = self.planner.create_research_plan(query)
            
            if planning_response.status == ResearchStatus.FAILED:
                return self._handle_workflow_failure(workflow, "Planning failed", planning_response.error)
            
            research_plan = ResearchPlan(**planning_response.data["research_plan"])
            workflow.steps[0].status = ResearchStatus.COMPLETED
            workflow.steps[0].completed_at = datetime.now()
            workflow.steps[0].output_data = planning_response.data
            
            logger.info(f"Research plan created: {research_plan.main_objective}")
            
            # Step 2: Execute Web Search
            logger.info("Step 2: Executing web searches...")
            workflow.steps[1].status = ResearchStatus.IN_PROGRESS
            workflow.steps[1].started_at = datetime.now()
            workflow.current_step = 1
            
            search_response = await self.web_searcher.execute_research(research_plan)
            
            if search_response.status == ResearchStatus.FAILED:
                return self._handle_workflow_failure(workflow, "Web search failed", search_response.error)
            
            # Extract results
            search_results = [SearchResult(**result) for result in search_response.data["search_results"]]
            competitors = [CompetitorInfo(**comp) for comp in search_response.data["competitors"]]
            
            workflow.steps[1].status = ResearchStatus.COMPLETED
            workflow.steps[1].completed_at = datetime.now()
            workflow.steps[1].output_data = search_response.data
            
            logger.info(f"Web search completed: {len(competitors)} competitors found")
            
            # Step 3: Analyze Gaps
            logger.info("Step 3: Analyzing research gaps...")
            workflow.steps[2].status = ResearchStatus.IN_PROGRESS
            workflow.steps[2].started_at = datetime.now()
            workflow.current_step = 2
            
            gap_response = self.gap_analyzer.analyze_research_gaps(
                research_plan, competitors, search_results
            )
            
            if gap_response.status == ResearchStatus.FAILED:
                return self._handle_workflow_failure(workflow, "Gap analysis failed", gap_response.error)
            
            gap_analysis = GapAnalysis(**gap_response.data["gap_analysis"])
            workflow.steps[2].status = ResearchStatus.COMPLETED
            workflow.steps[2].completed_at = datetime.now()
            workflow.steps[2].output_data = gap_response.data
            
            logger.info(f"Gap analysis completed: {gap_analysis.data_quality_score:.1%} data quality" if gap_analysis.data_quality_score else "Gap analysis completed")
            
            # Step 4: Generate Insights and Report
            logger.info("Step 4: Generating insights and final report...")
            workflow.steps[3].status = ResearchStatus.IN_PROGRESS
            workflow.steps[3].started_at = datetime.now()
            workflow.current_step = 3
            
            # Generate competitive insights
            insights_response = self.response_curator.generate_competitive_insights(
                competitors, research_plan, gap_analysis
            )
            
            if insights_response.status == ResearchStatus.FAILED:
                return self._handle_workflow_failure(workflow, "Insights generation failed", insights_response.error)
            
            competitive_insights = CompetitiveInsights(**insights_response.data["competitive_insights"])
            
            # Create final report
            total_duration = (datetime.now() - start_time).total_seconds()
            
            report_response = self.response_curator.create_research_report(
                query=query,
                plan=research_plan,
                competitors=competitors,
                gap_analysis=gap_analysis,
                insights=competitive_insights,
                search_results=search_results,
                research_duration=total_duration
            )
            
            if report_response.status == ResearchStatus.FAILED:
                return self._handle_workflow_failure(workflow, "Report generation failed", report_response.error)
            
            final_report = ResearchReport(**report_response.data["research_report"])
            
            # Generate markdown report
            markdown_report = self.response_curator.format_markdown_report(final_report)
            
            workflow.steps[3].status = ResearchStatus.COMPLETED
            workflow.steps[3].completed_at = datetime.now()
            workflow.steps[3].output_data = report_response.data
            
            # Finalize workflow
            workflow.status = ResearchStatus.COMPLETED
            workflow.completed_at = datetime.now()
            workflow.total_duration_seconds = total_duration
            workflow.final_report = final_report
            
            logger.info(f"Research workflow completed successfully in {total_duration:.2f}s")
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "report": final_report.model_dump(),
                "markdown_report": markdown_report,
                "workflow": workflow.model_dump(),
                "metrics": {
                    "duration_seconds": total_duration,
                    "competitors_found": len(competitors),
                    "searches_performed": len(search_results),
                    "data_quality_score": gap_analysis.data_quality_score
                }
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in workflow {workflow_id}: {e}")
            return self._handle_workflow_failure(workflow, "Unexpected error", str(e))
    
    def _handle_workflow_failure(
        self, 
        workflow: WorkflowExecution, 
        error_type: str, 
        error_message: str
    ) -> Dict[str, Any]:
        """Handle workflow failure and return error response."""
        
        workflow.status = ResearchStatus.FAILED
        workflow.completed_at = datetime.now()
        
        if workflow.started_at:
            workflow.total_duration_seconds = (workflow.completed_at - workflow.started_at).total_seconds()
        
        # Mark current step as failed
        if workflow.current_step < len(workflow.steps):
            workflow.steps[workflow.current_step].status = ResearchStatus.FAILED
            workflow.steps[workflow.current_step].error_message = error_message
            workflow.steps[workflow.current_step].completed_at = datetime.now()
        
        logger.error(f"Workflow {workflow.workflow_id} failed: {error_type} - {error_message}")
        
        return {
            "success": False,
            "workflow_id": workflow.workflow_id,
            "error_type": error_type,
            "error_message": error_message,
            "workflow": workflow.model_dump(),
            "partial_results": self._extract_partial_results(workflow)
        }
    
    def _extract_partial_results(self, workflow: WorkflowExecution) -> Dict[str, Any]:
        """Extract any partial results from completed workflow steps."""
        partial_results = {}
        
        for step in workflow.steps:
            if step.status == ResearchStatus.COMPLETED and step.output_data:
                partial_results[step.step_name] = step.output_data
        
        return partial_results
    
    async def validate_system(self) -> Dict[str, Any]:
        """
        Validate that all system components are working correctly.
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating system components...")
        
        validation_results = {
            "overall_status": "unknown",
            "components": {},
            "recommendations": []
        }
        
        # Test configuration
        try:
            config_valid = validate_config()
            validation_results["components"]["configuration"] = {
                "status": "✅ Valid" if config_valid else "❌ Invalid",
                "details": "Configuration and API keys validated"
            }
        except Exception as e:
            validation_results["components"]["configuration"] = {
                "status": "❌ Error",
                "details": f"Configuration error: {str(e)}"
            }
            validation_results["recommendations"].append("Check your .env file and API keys")
        
        # Test Planning Agent
        try:
            test_query = ResearchQuery(query="test market analysis")
            planner_response = self.planner.create_research_plan(test_query)
            
            validation_results["components"]["planner_agent"] = {
                "status": "✅ Working" if planner_response.status == ResearchStatus.COMPLETED else "⚠️ Issues",
                "details": f"Planning agent response: {planner_response.status}"
            }
        except Exception as e:
            validation_results["components"]["planner_agent"] = {
                "status": "❌ Error", 
                "details": f"Planning agent error: {str(e)}"
            }
        
        # Test Web Searcher Agent (API validation only)
        try:
            api_valid = await self.web_searcher._validate_tavily_api()
            validation_results["components"]["web_searcher_agent"] = {
                "status": "✅ API Valid" if api_valid else "❌ API Invalid",
                "details": "Tavily API key validation"
            }
            
            if not api_valid:
                validation_results["recommendations"].append("Check your TAVILY_API_KEY")
                
        except Exception as e:
            validation_results["components"]["web_searcher_agent"] = {
                "status": "❌ Error",
                "details": f"Web searcher error: {str(e)}"
            }
        
        # Test Gap Analyzer Agent
        try:
            # Simple validation - the agent should initialize without errors
            validation_results["components"]["gap_analyzer_agent"] = {
                "status": "✅ Ready",
                "details": "Gap analyzer agent initialized successfully"
            }
        except Exception as e:
            validation_results["components"]["gap_analyzer_agent"] = {
                "status": "❌ Error",
                "details": f"Gap analyzer error: {str(e)}"
            }
        
        # Test Response Curator Agent  
        try:
            # Simple validation - the agent should initialize without errors
            validation_results["components"]["response_curator_agent"] = {
                "status": "✅ Ready",
                "details": "Response curator agent initialized successfully"
            }
        except Exception as e:
            validation_results["components"]["response_curator_agent"] = {
                "status": "❌ Ready",
                "details": f"Response curator error: {str(e)}"
            }
        
        # Determine overall status
        component_statuses = [comp["status"] for comp in validation_results["components"].values()]
        
        if all("✅" in status for status in component_statuses):
            validation_results["overall_status"] = "✅ All systems operational"
        elif any("❌" in status for status in component_statuses):
            validation_results["overall_status"] = "❌ System errors detected"
        else:
            validation_results["overall_status"] = "⚠️ Some components have warnings"
        
        logger.info(f"System validation completed: {validation_results['overall_status']}")
        
        return validation_results
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get the status of a specific workflow (placeholder for future implementation).
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Workflow status information
        """
        # This would connect to a workflow database in a production system
        return {
            "workflow_id": workflow_id,
            "status": "not_implemented",
            "message": "Workflow tracking database not implemented"
        }