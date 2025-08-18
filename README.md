# CDR RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline for analyzing Call Detail Records (CDR) data using Mistral LLM via Ollama and ChromaDB for vector storage.

## Features

- **CDR Data Processing**: Parse and analyze telecommunications call records
- **Semantic Search**: Vector-based search using sentence transformers
- **LLM Analysis**: Mistral model integration via Ollama for intelligent insights
- **Dynamic Analytics**: Automatic chart generation and pattern analysis
- **Web Interface**: HTML dashboard with interactive charts
- **FastAPI Backend**: High-performance API with automatic documentation

## Architecture

```
CDR JSON → Parse & Extract → Create Documents → Generate Embeddings → Store in ChromaDB
                                                                            ↓
User Query → Query Embedding → Similarity Search → Context Building → Mistral LLM → Response + Charts
```

## Quick Start

1. **Install Ollama**

   For macOS:
   - Visit https://ollama.ai/download
   - Download "Ollama for macOS"
   - Install the .app file
   - Launch Ollama from Applications
   
   Or via Homebrew:
   ```bash
   brew install ollama
   ```
   
   For Linux:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Run the setup script**
   ```bash
   ./start.sh
   ```

The script will:
- Create a Python virtual environment
- Install all dependencies
- Download the Mistral model
- Start the web interface at http://localhost:8000

## Usage

### Web Interface

After running `./start.sh`, open your browser to:
```
http://localhost:8000
```

Example queries for your CDR data:
- "How many calls were made today?"
- "Show me calls between extension 5656 and 5651"
- "What's the calling pattern by hour?"
- "Find calls longer than 30 seconds"

### API Endpoints

- `GET /` - Web dashboard
- `POST /query` - Query the RAG pipeline
- `GET /health` - Health check
- `GET /stats` - Data statistics
- `GET /docs` - API documentation

## Data Format

The pipeline expects CDR data in Elasticsearch export format (JSON). Your `es_data.json` should contain:

```json
{
  "hits": {
    "hits": [
      {
        "_source": {
          "fields": {
            "communicationRecord": {
              "startTime": "2025-06-12T11:24:16.000Z",
              "endTime": "2025-06-12T11:24:17.000Z",
              "direction": "outgoing",
              "comType": "VoiceCall"
            },
            "participants": [
              {"address": "5656", "involvement": "source"},
              {"address": "5651", "involvement": "destination"}
            ],
            "deviceInfo": {
              "pUserID": "user123",
              "audioInterface": "Handset 1"
            }
          }
        }
      }
    ]
  }
}
```

## Query Examples

### Basic Queries
- "How many calls are in the dataset?"
- "What's the total call duration?"
- "Show me the most active participants"

### Time-based Queries
- "What's the calling pattern by hour of day?"
- "Find calls made yesterday"

### Participant Queries
- "Show me calls between 5656 and 5651"
- "Find all calls made by user ci"
- "Who does extension 5651 call most?"

### Pattern Analysis
- "Find unusual calling patterns"
- "Show me the longest calls"
- "What's the ratio of incoming vs outgoing calls?"

## Configuration

Edit `config/settings.py` to customize:

- **Data paths**: Location of your CDR data
- **Model settings**: Embedding model and LLM configuration
- **ChromaDB settings**: Vector database configuration
- **API settings**: Server host and port


### Project Structure
```
ragproject/
├── start.sh               # Setup and startup script
├── app.py                 # FastAPI web application
├── main.py                # Main RAG pipeline
├── rag_components.py      # Data processing utilities
├── model_setup.py         # LLM and embedding model setup
├── chroma_setup.py        # ChromaDB configuration
├── config/
│   └── settings.py        # Configuration settings
├── templates/
│   └── dashboard.html     # Web interface
├── data/
│   └── es_data.json       # CDR data
├── venv/                  # Python virtual environment
└── requirements.txt       # Python dependencies
```


