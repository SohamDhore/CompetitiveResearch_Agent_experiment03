"""
Main CLI application for the Competitive Research Multi-Agent System.
Provides a command-line interface for executing competitive research.
"""

import asyncio
import argparse
import logging
import json
import os
import sys
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from config import validate_config, get_config_summary
from models import ResearchQuery, ResearchDepth
from orchestrator import OrchestratorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
console = Console()

class CompetitiveResearchCLI:
    """CLI application for competitive research."""
    
    def __init__(self):
        """Initialize the CLI application."""
        self.orchestrator = None
        
    async def initialize(self):
        """Initialize the orchestrator and validate system."""
        try:
            console.print("\n🚀 [bold blue]Initializing Competitive Research Multi-Agent System[/bold blue]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                
                # Validate configuration
                task = progress.add_task("Validating configuration...", total=None)
                if not validate_config():
                    console.print("❌ [red]Configuration validation failed[/red]")
                    console.print("\nPlease check:")
                    console.print("- Your .env file exists and has correct API keys")
                    console.print("- OPENAI_API_KEY is set")
                    console.print("- TAVILY_API_KEY is set")
                    console.print("- Model is set to gpt-5-mini")
                    return False
                
                # Initialize orchestrator
                progress.update(task, description="Initializing orchestrator...")
                self.orchestrator = OrchestratorAgent()
                
                # Validate system components
                progress.update(task, description="Validating system components...")
                validation_results = await self.orchestrator.validate_system()
                
                progress.stop()
                
                # Display validation results
                self._display_validation_results(validation_results)
                
                return "✅" in validation_results["overall_status"]
                
        except Exception as e:
            console.print(f"❌ [red]Initialization failed: {str(e)}[/red]")
            logger.error(f"Initialization error: {e}")
            return False
    
    def _display_validation_results(self, results):
        """Display system validation results."""
        console.print("\n📋 [bold]System Validation Results[/bold]")
        
        # Create validation table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        for component, info in results["components"].items():
            table.add_row(
                component.replace("_", " ").title(),
                info["status"],
                info["details"]
            )
        
        console.print(table)
        
        # Overall status
        status_color = "green" if "✅" in results["overall_status"] else "red"
        console.print(f"\n🎯 [bold {status_color}]{results['overall_status']}[/bold {status_color}]")
        
        # Recommendations
        if results["recommendations"]:
            console.print("\n💡 [bold yellow]Recommendations:[/bold yellow]")
            for rec in results["recommendations"]:
                console.print(f"  • {rec}")
    
    async def run_research(self, query_text: str, depth: str = "standard", 
                          focus_areas: Optional[list] = None, 
                          save_report: bool = True) -> bool:
        """
        Run competitive research analysis.
        
        Args:
            query_text: Research query
            depth: Research depth (basic, standard, comprehensive)
            focus_areas: Specific areas to focus on
            save_report: Whether to save the report to file
            
        Returns:
            True if research completed successfully
        """
        try:
            # Create research query
            query = ResearchQuery(
                query=query_text,
                depth=ResearchDepth(depth),
                focus_areas=focus_areas or [],
                max_results=10
            )
            
            console.print(f"\n🔍 [bold green]Starting Research:[/bold green] {query.query}")
            console.print(f"📊 [blue]Depth:[/blue] {query.depth.value}")
            if focus_areas:
                console.print(f"🎯 [blue]Focus Areas:[/blue] {', '.join(focus_areas)}")
            
            # Execute research with progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                
                task = progress.add_task("Executing research workflow...", total=None)
                
                # Run the research
                results = await self.orchestrator.execute_research(query)
                
                progress.stop()
            
            # Display results
            if results["success"]:
                await self._display_research_results(results)
                
                if save_report:
                    report_path = await self._save_research_report(results)
                    if report_path:
                        console.print(f"\n💾 [green]Report saved to:[/green] {report_path}")
                
                return True
            else:
                console.print(f"\n❌ [red]Research failed:[/red] {results['error_message']}")
                
                # Display any partial results
                if results.get("partial_results"):
                    console.print("\n📋 [yellow]Partial Results Available:[/yellow]")
                    for step, data in results["partial_results"].items():
                        console.print(f"  • {step.replace('_', ' ').title()}: ✅ Completed")
                
                return False
                
        except Exception as e:
            console.print(f"❌ [red]Research execution failed: {str(e)}[/red]")
            logger.error(f"Research execution error: {e}")
            return False
    
    async def _display_research_results(self, results):
        """Display research results in a formatted way."""
        metrics = results["metrics"]
        report = results["report"]
        
        # Display metrics
        console.print("\n📊 [bold blue]Research Metrics[/bold blue]")
        
        metrics_table = Table(show_header=False, box=None)
        metrics_table.add_column("Metric", style="cyan", no_wrap=True)
        metrics_table.add_column("Value", style="green")
        
        metrics_table.add_row("Duration", f"{metrics['duration_seconds']:.1f} seconds")
        metrics_table.add_row("Competitors Found", str(metrics['competitors_found']))
        metrics_table.add_row("Searches Performed", str(metrics['searches_performed']))
        
        if metrics.get('data_quality_score'):
            metrics_table.add_row("Data Quality", f"{metrics['data_quality_score']:.1%}")
        
        console.print(metrics_table)
        
        # Display executive summary
        if report.get('executive_summary'):
            console.print(f"\n📄 [bold blue]Executive Summary[/bold blue]")
            summary_panel = Panel(report['executive_summary'], expand=False)
            console.print(summary_panel)
        
        # Display top competitors
        competitors = report.get('competitors', [])[:5]  # Show top 5
        if competitors:
            console.print(f"\n🏢 [bold blue]Top Competitors[/bold blue]")
            
            comp_table = Table(show_header=True, header_style="bold magenta")
            comp_table.add_column("Company", style="cyan", no_wrap=True)
            comp_table.add_column("Website", style="blue")
            comp_table.add_column("Products", style="green")
            
            for comp in competitors:
                comp_table.add_row(
                    comp['name'],
                    comp.get('website', 'N/A')[:50] + '...' if comp.get('website') and len(comp.get('website', '')) > 50 else comp.get('website', 'N/A'),
                    ', '.join(comp.get('products', [])[:2])
                )
            
            console.print(comp_table)
        
        # Display key insights
        insights = report.get('insights', {})
        if insights:
            console.print(f"\n💡 [bold blue]Key Insights[/bold blue]")
            
            # Market opportunities
            opportunities = insights.get('market_opportunities', [])[:3]
            if opportunities:
                console.print("\n🎯 [bold green]Market Opportunities:[/bold green]")
                for opp in opportunities:
                    console.print(f"  • {opp}")
            
            # Strategic recommendations
            recommendations = insights.get('strategic_recommendations', [])[:3]
            if recommendations:
                console.print("\n📈 [bold yellow]Strategic Recommendations:[/bold yellow]")
                for rec in recommendations:
                    console.print(f"  • {rec}")
    
    async def _save_research_report(self, results) -> Optional[str]:
        """Save research report to file."""
        try:
            # Create reports directory
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            query_slug = results["report"]["query"]["query"][:30].replace(" ", "_").replace("/", "_")
            filename = f"competitive_research_{query_slug}_{timestamp}.md"
            filepath = os.path.join(reports_dir, filename)
            
            # Save markdown report
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(results["markdown_report"])
            
            # Also save JSON data
            json_filename = filename.replace('.md', '_data.json')
            json_filepath = os.path.join(reports_dir, json_filename)
            
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(results["report"], f, indent=2, default=str)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            console.print(f"⚠️ [yellow]Could not save report: {str(e)}[/yellow]")
            return None
    
    def show_config(self):
        """Display current configuration."""
        console.print("\n⚙️ [bold blue]Current Configuration[/bold blue]")
        
        config = get_config_summary()
        
        config_table = Table(show_header=True, header_style="bold magenta")
        config_table.add_column("Setting", style="cyan", no_wrap=True)
        config_table.add_column("Value", style="green")
        
        for key, value in config.items():
            if key == "api_keys_configured":
                for api_key, configured in value.items():
                    status = "✅ Configured" if configured else "❌ Missing"
                    config_table.add_row(f"{api_key.upper()}_API_KEY", status)
            else:
                config_table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(config_table)

async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Competitive Research Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py research "AI chatbot companies"
  python main.py research "cloud storage competitors" --depth comprehensive
  python main.py research "email marketing tools" --focus pricing features
  python main.py validate
  python main.py config
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Research command
    research_parser = subparsers.add_parser("research", help="Execute competitive research")
    research_parser.add_argument("query", help="Research query or question")
    research_parser.add_argument(
        "--depth", 
        choices=["basic", "standard", "comprehensive"],
        default="standard",
        help="Research depth level (default: standard)"
    )
    research_parser.add_argument(
        "--focus",
        nargs="+",
        help="Focus areas (e.g., pricing features market_position)"
    )
    research_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the research report to file"
    )
    
    # Validate command
    subparsers.add_parser("validate", help="Validate system components")
    
    # Config command
    subparsers.add_parser("config", help="Show current configuration")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = CompetitiveResearchCLI()
    
    # Initialize system for all commands except config
    if args.command != "config":
        initialized = await cli.initialize()
        if not initialized:
            console.print("\n❌ [red]System initialization failed. Cannot proceed.[/red]")
            return
    
    try:
        if args.command == "research":
            success = await cli.run_research(
                query_text=args.query,
                depth=args.depth,
                focus_areas=args.focus,
                save_report=not args.no_save
            )
            sys.exit(0 if success else 1)
            
        elif args.command == "validate":
            validation_results = await cli.orchestrator.validate_system()
            console.print("\n✅ [green]System validation completed.[/green]")
            sys.exit(0 if "✅" in validation_results["overall_status"] else 1)
            
        elif args.command == "config":
            cli.show_config()
            
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        console.print("\n\n⚠️ [yellow]Research interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ [red]Unexpected error: {str(e)}[/red]")
        logger.error(f"Unexpected error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())