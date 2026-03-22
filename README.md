# Multi-Agent Legal Workflow

> Multi-Agent Legal Workflow System built with LangGraph and CrewAI

## Architecture

![Architecture Diagram](architecture/architecture-diagram.png)

<!-- TODO: Replace with actual architecture diagram from excalidraw.com -->

## Overview

A multi-agent AI system that orchestrates specialized legal workflow agents using LangGraph's stateful graph execution. Each agent handles a distinct legal task — contract review, clause extraction, compliance checking, and summarization — coordinated through a supervisor agent for end-to-end legal document processing.

### Key Features

- **Supervisor Agent**: Orchestrates task delegation and workflow routing via LangGraph
- **Contract Review Agent**: Analyzes contracts for key terms, obligations, and deadlines
- **Compliance Agent**: Checks clauses against regulatory requirements
- **Summarization Agent**: Generates executive summaries with risk highlights
- **Human-in-the-Loop**: Configurable approval gates for high-risk decisions

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | LangGraph, CrewAI |
| LLM | AWS Bedrock (Claude 3.5 Sonnet) |
| Orchestration | LangChain |
| State Management | LangGraph Checkpointer |
| Backend | FastAPI |
| Frontend | Streamlit |
| Monitoring | LangSmith |

## Setup

### Prerequisites

- Python 3.11+
- AWS account with Bedrock access
- LangSmith account (optional, for tracing)

### Installation

```bash
git clone https://github.com/gyaneshwarmalba/multi-agent-legal-workflow.git
cd multi-agent-legal-workflow
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Fill in your credentials
```

### Running

```bash
streamlit run app/main.py
```

## Demo

<!-- TODO: Add demo GIF/video -->

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
