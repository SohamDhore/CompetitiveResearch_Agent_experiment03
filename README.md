# Competitive Research Multi-Agent System

A sophisticated multi-agent system for competitive intelligence and market research, powered by **GPT-5-mini** and **Tavily AI**.

## 🎯 Overview

This system uses an orchestrated multi-agent approach to conduct comprehensive competitive research:

- **🎯 Orchestrator Agent**: Coordinates the entire research workflow
- **📋 Planning Agent**: Creates strategic research plans and queries (GPT-5-mini)
- **🔍 Web Search Agent**: Uses Tavily AI for intelligent web search (GPT-5-mini)
- **📊 Gap Analysis Agent**: Identifies research gaps and missing information (GPT-5-mini)
- **📝 Response Curator Agent**: Synthesizes findings into professional reports (GPT-5-mini)

## ✨ Key Features

- **AI-Powered Research**: GPT-5-mini for all analysis and planning
- **Specialized Web Search**: Tavily AI for comprehensive market intelligence
- **Automated Workflow**: Orchestrated agent coordination
- **Professional Reports**: Markdown reports with strategic insights
- **Gap Analysis**: Identifies missing information and suggests improvements
- **CLI Interface**: Easy-to-use command-line interface
- **Type Safety**: Pydantic models for data validation

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for GPT-5-mini)
- Tavily AI API key

### Installation

1. **Clone or create the project directory:**
   ```bash
   cd "competitor 3"
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

### Usage

#### Basic Research
```bash
python main.py research "AI chatbot companies"
```

#### Comprehensive Research
```bash
python main.py research "cloud storage competitors" --depth comprehensive
```

#### Focused Research
```bash
python main.py research "email marketing tools" --focus pricing features market_position
```

#### System Validation
```bash
python main.py validate
```

#### View Configuration
```bash
python main.py config
```

## 📊 Research Workflow

1. **Planning Phase**: Creates strategic research plan with key questions and search keywords
2. **Search Phase**: Executes intelligent web searches using Tavily AI
3. **Analysis Phase**: Identifies gaps and missing information  
4. **Curation Phase**: Generates comprehensive reports with strategic insights

## 🏗️ Architecture

### Agent Structure

```
OrchestratorAgent
├── PlannerAgent (GPT-4o-mini)
├── WebSearcherAgent (Tavily AI + GPT-4o-mini)
├── GapAnalyzerAgent (GPT-4o-mini)
└── ResponseCuratorAgent (GPT-4o-mini)
```

### Data Models

- **ResearchQuery**: Input query specification
- **ResearchPlan**: Strategic research strategy
- **CompetitorInfo**: Structured competitor data
- **GapAnalysis**: Research completeness analysis
- **ResearchReport**: Final comprehensive report

## 📈 Output Features

- **Executive Summaries**: Strategic overview of findings
- **Competitor Profiles**: Detailed company information
- **Market Analysis**: Opportunities, threats, and recommendations
- **Gap Analysis**: Data quality scores and missing information
- **Strategic Recommendations**: Actionable business insights
- **Professional Formatting**: Clean markdown reports with citations

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `TAVILY_API_KEY` | Tavily AI API key (required) | - |
| `OPENAI_MODEL` | LLM model | `gpt-5-mini` |
| `TEMPERATURE` | Model temperature | `1.0` |
| `MAX_SEARCH_RESULTS` | Results per search | `10` |
| `TAVILY_SEARCH_DEPTH` | Search depth | `advanced` |
| `REQUEST_TIMEOUT` | Request timeout (seconds) | `30` |

### Research Depth Levels

- **Basic**: 3-5 competitors, essential information
- **Standard**: 5-8 competitors, comprehensive analysis  
- **Comprehensive**: 8-12 competitors, deep market analysis

## 📁 Project Structure

```
competitor 3/
├── main.py                    # CLI entry point
├── config.py                  # Configuration management
├── models.py                  # Pydantic data models
├── orchestrator.py            # Master coordinator
├── agents/
│   ├── __init__.py
│   ├── planner_agent.py       # Research planning
│   ├── web_searcher_agent.py  # Tavily AI integration
│   ├── gap_analyzer_agent.py  # Gap analysis
│   └── response_curator_agent.py # Report generation
├── reports/                   # Generated reports
├── requirements.txt           # Dependencies
├── .env.example              # Environment template
└── README.md                 # Documentation
```

## 🔍 Example Commands

### Market Research
```bash
# Research SaaS competitors
python main.py research "SaaS productivity tools" --depth comprehensive

# Focus on specific areas
python main.py research "video conferencing software" --focus pricing features security

# Quick competitor overview
python main.py research "project management tools" --depth basic
```

### System Commands
```bash
# Validate all components
python main.py validate

# Check configuration
python main.py config

# View help
python main.py --help
python main.py research --help
```

## 📊 Sample Output

The system generates comprehensive reports including:

- **Executive Summary**: Strategic overview
- **Competitive Landscape**: Detailed competitor profiles
- **Strategic Analysis**: Market opportunities and threats
- **Gap Analysis**: Data quality and missing information
- **Methodology**: Research approach and limitations
- **Next Steps**: Recommended actions

Reports are saved in both Markdown (`.md`) and JSON (`.json`) formats in the `reports/` directory.

## 🔧 Development

### Running Tests
```bash
pip install pytest pytest-asyncio
pytest
```

### Code Formatting
```bash
pip install black flake8
black .
flake8 .
```

### Logging
- Logs are written to `research.log`
- Set `LOG_LEVEL` environment variable to control verbosity
- Use `LOG_FILE` to specify custom log file location

## 🚨 Error Handling

The system includes comprehensive error handling:

- **API Failures**: Automatic retry with exponential backoff
- **Network Issues**: Timeout handling and graceful degradation
- **Data Validation**: Pydantic model validation with fallbacks
- **Partial Results**: Returns available data even if some steps fail

## 🔐 Security

- **API Keys**: Stored in environment variables, never committed
- **Input Validation**: All inputs validated through Pydantic models
- **Rate Limiting**: Respects API rate limits with intelligent backoff
- **Error Sanitization**: Sensitive information filtered from logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:

1. Check the logs in `research.log`
2. Run `python main.py validate` to check system status
3. Verify API keys in `.env` file
4. Ensure Python 3.8+ is installed

## 🔗 API Documentation

- **OpenAI GPT-5-mini**: https://platform.openai.com/docs
- **Tavily AI**: https://docs.tavily.com/
- **Pydantic**: https://docs.pydantic.dev/

---

*Powered by GPT-5-mini and Tavily AI for next-generation competitive intelligence.*