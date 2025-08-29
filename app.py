"""
FastAPI Application for CDR RAG Pipeline
Web interface for querying Call Detail Records using RAG
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from main import rag_pipeline, filter_dataframe_by_query, analyze_query_intent
from rag_components import load_es_data, prepare_documents_for_chroma_and_df
from model_setup import initialize_models
from chroma_setup import initialize_chromadb
from config.settings import ES_DATA_FILE, API_HOST, API_PORT
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CDR RAG Pipeline API",
    description="Retrieval-Augmented Generation pipeline for Call Detail Records analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = project_root / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

class QueryRequest(BaseModel):
    query: str

class HealthResponse(BaseModel):
    status: str
    message: str
    components: dict

# Global objects for pipeline (initialized on startup)
es_raw_hits = None
documents = None
metadatas = None
ids = None
df_all_records = None
embedding_model = None
llm_pipeline = None
embedding_model_name = None
chroma_collection = None

# Simple in-memory cache for LLM and retrieval results
query_cache = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline components on startup"""
    global es_raw_hits, documents, metadatas, ids, df_all_records
    global embedding_model, llm_pipeline, embedding_model_name, chroma_collection
    
    logger.info("Initializing CDR RAG Pipeline...")
    
    try:
        # 1. Load CDR data
        logger.info("Loading CDR data...")
        es_raw_hits = load_es_data(str(ES_DATA_FILE))
        
        if not es_raw_hits:
            raise Exception(f"No CDR data found in {ES_DATA_FILE}")
        
        # 2. Prepare documents
        logger.info("Preparing documents for ChromaDB...")
        documents, metadatas, ids, df_all_records = prepare_documents_for_chroma_and_df(es_raw_hits)
        
        # 3. Initialize models
        logger.info("Initializing embedding and LLM models...")
        embedding_model, llm_pipeline, _, embedding_model_name = initialize_models()
        
        # 4. Initialize ChromaDB
        logger.info("Initializing ChromaDB collection...")
        chroma_collection = initialize_chromadb(
            documents, metadatas, ids, embedding_model, embedding_model_name
        )
        
        logger.info("CDR RAG Pipeline initialized successfully!")
        logger.info(f"Loaded {len(documents)} CDR documents")
        logger.info(f"DataFrame shape: {df_all_records.shape}")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the main dashboard HTML"""
    html_file = project_root / "templates" / "dashboard.html"
    
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(), status_code=200)
    else:
        # Return a simple HTML page if template doesn't exist
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CDR RAG Pipeline</title>
        </head>
        <body>
            <h1>CDR RAG Pipeline API</h1>
            <p>The RAG pipeline is running. Use the /query endpoint to ask questions about CDR data.</p>
            <p>API Documentation: <a href="/docs">/docs</a></p>
        </body>
        </html>
        """, status_code=200)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global chroma_collection, embedding_model, llm_pipeline, df_all_records
    
    components = {
        "cdr_data_loaded": es_raw_hits is not None and len(es_raw_hits) > 0,
        "documents_prepared": documents is not None and len(documents) > 0,
        "embedding_model": embedding_model is not None,
        "llm_model": llm_pipeline is not None,
        "chromadb_collection": chroma_collection is not None,
        "dataframe_ready": df_all_records is not None and not df_all_records.empty
    }
    
    all_healthy = all(components.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        message="All components operational" if all_healthy else "Some components not ready",
        components=components
    )

@app.get("/stats")
async def get_stats():
    """Get statistics about the loaded CDR data"""
    global df_all_records, chroma_collection
    
    if df_all_records is None or df_all_records.empty:
        raise HTTPException(status_code=503, detail="CDR data not loaded")
    
    try:
        stats = {
            "total_records": len(df_all_records),
            "date_range": {
                "start": df_all_records['start_time'].min() if 'start_time' in df_all_records.columns else None,
                "end": df_all_records['start_time'].max() if 'start_time' in df_all_records.columns else None
            },
            "unique_participants": {
                "sources": df_all_records['source_address'].nunique() if 'source_address' in df_all_records.columns else 0,
                "destinations": df_all_records['destination_address'].nunique() if 'destination_address' in df_all_records.columns else 0
            },
            "call_directions": df_all_records['direction'].value_counts().to_dict() if 'direction' in df_all_records.columns else {},
            "platforms": df_all_records['platform'].value_counts().to_dict() if 'platform' in df_all_records.columns else {},
            "avg_duration": df_all_records['duration_seconds'].mean() if 'duration_seconds' in df_all_records.columns else 0,
            "total_duration": df_all_records['duration_seconds'].sum() if 'duration_seconds' in df_all_records.columns else 0
        }
        
        if chroma_collection:
            stats["chromadb_documents"] = chroma_collection.count()
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")

@app.post("/query")
async def handle_query(data: QueryRequest):
    """Handle user queries using the RAG pipeline"""
    global df_all_records, chroma_collection, llm_pipeline, embedding_model, metadatas
    
    query = data.query.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Check if components are ready
    if not all([df_all_records is not None, chroma_collection is not None, 
                llm_pipeline is not None, embedding_model is not None]):
        raise HTTPException(status_code=503, detail="RAG pipeline not fully initialized")
    
    logger.info(f"[API] Received query: {query}")
    
    # Check cache first
    if query in query_cache:
        logger.info("[API] Returning cached result")
        return query_cache[query]
    
    try:
        logger.info("[API] Calling rag_pipeline...")
        
        answer = rag_pipeline(
            query=query,
            df_all_records=df_all_records,
            chroma_collection=chroma_collection,
            llm_client=llm_pipeline,
            embedding_model=embedding_model,
            metadatas=metadatas
        )
        
        logger.info(f"[API] Pipeline completed. Response keys: {list(answer.keys())}")
        
        # Cache the result
        query_cache[query] = answer
        
        # Limit cache size
        if len(query_cache) > 100:
            # Remove oldest entry
            oldest_key = next(iter(query_cache))
            del query_cache[oldest_key]
        
        return answer
        
    except Exception as e:
        logger.error(f"[API] Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/clear-cache")
async def clear_cache():
    """Clear the query cache"""
    global query_cache
    
    cache_size = len(query_cache)
    query_cache.clear()
    
    return {
        "message": f"Cache cleared. Removed {cache_size} entries.",
        "cache_size": len(query_cache)
    }

@app.post("/graph-data")
async def get_graph_data(data: QueryRequest):
    """Generate graph data for count queries"""
    global df_all_records
    
    query = data.query.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if df_all_records is None or df_all_records.empty:
        raise HTTPException(status_code=503, detail="CDR data not loaded")
    
    logger.info(f"[GRAPH] Received query for graph data: {query}")
    
    try:
        # Analyze query intent
        query_intent = analyze_query_intent(query)
        
        if not (query_intent.get('is_count_query', False) or query_intent.get('requires_chart', False)):
            raise HTTPException(status_code=400, detail="Query is not suitable for graph generation. Graphs are generated for count queries and temporal analysis.")
        
        filtered_df = filter_dataframe_by_query(df_all_records, query)
        
        if filtered_df.empty:
            return {
                "chart_type": "empty",
                "data": [],
                "labels": [],
                "title": "No data found for the query",
                "query": query
            }
        
        # Convert start_time to datetime if needed
        filtered_df = filtered_df.copy()
        filtered_df['start_time'] = pd.to_datetime(filtered_df['start_time'], errors='coerce')
        
        # Determine chart type and data based on query
        chart_data = None
        
        # Daily breakdown
        if "day" in query.lower() or "daily" in query.lower():
            daily_counts = filtered_df.groupby(filtered_df['start_time'].dt.date).size()
            chart_data = {
                "chart_type": "bar",
                "data": daily_counts.values.tolist(),
                "labels": [str(date) for date in daily_counts.index],
                "title": f"Daily Call Counts",
                "query": query
            }
        
        # Monthly breakdown
        elif "month" in query.lower() or "monthly" in query.lower():
            monthly_counts = filtered_df.groupby(filtered_df['start_time'].dt.to_period('M')).size()
            chart_data = {
                "chart_type": "bar",
                "data": monthly_counts.values.tolist(),
                "labels": [str(month) for month in monthly_counts.index],
                "title": f"Monthly Call Counts",
                "query": query
            }
        
        # Yearly breakdown
        elif "year" in query.lower() or "yearly" in query.lower():
            yearly_counts = filtered_df.groupby(filtered_df['start_time'].dt.year).size()
            chart_data = {
                "chart_type": "bar",
                "data": yearly_counts.values.tolist(),
                "labels": [str(year) for year in yearly_counts.index],
                "title": f"Yearly Call Counts",
                "query": query
            }
        
        # Hourly breakdown (for specific days)
        elif any(word in query.lower() for word in ["hour", "hourly", "time"]):
            hourly_counts = filtered_df.groupby(filtered_df['start_time'].dt.hour).size()
            chart_data = {
                "chart_type": "line",
                "data": hourly_counts.values.tolist(),
                "labels": [f"{hour}:00" for hour in hourly_counts.index],
                "title": f"Hourly Call Distribution",
                "query": query
            }
        
        # Direction breakdown (incoming vs outgoing)
        elif "incoming" in query.lower() or "outgoing" in query.lower() or "direction" in query.lower():
            direction_counts = filtered_df['direction'].value_counts()
            chart_data = {
                "chart_type": "pie",
                "data": direction_counts.values.tolist(),
                "labels": direction_counts.index.tolist(),
                "title": f"Call Direction Distribution",
                "query": query
            }
        
        # Platform breakdown
        elif "platform" in query.lower():
            platform_counts = filtered_df['platform'].value_counts()
            chart_data = {
                "chart_type": "pie",
                "data": platform_counts.values.tolist(),
                "labels": platform_counts.index.tolist(),
                "title": f"Platform Distribution",
                "query": query
            }
        
        # Default: total count with single bar
        else:
            total_count = len(filtered_df)
            chart_data = {
                "chart_type": "bar",
                "data": [total_count],
                "labels": ["Total Calls"],
                "title": f"Total Call Count",
                "query": query
            }
        
        # Add metadata
        chart_data["total_records"] = len(filtered_df)
        chart_data["date_range"] = {
            "start": str(filtered_df['start_time'].min().date()) if not filtered_df.empty else None,
            "end": str(filtered_df['start_time'].max().date()) if not filtered_df.empty else None
        }
        
        logger.info(f"[GRAPH] Generated {chart_data['chart_type']} chart with {len(chart_data['data'])} data points")
        
        return chart_data
        
    except Exception as e:
        logger.error(f"[GRAPH] Error generating graph data: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating graph data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting CDR RAG Pipeline API server...")
    logger.info(f"Data file: {ES_DATA_FILE}")
    logger.info(f"Server will run at: http://{API_HOST}:{API_PORT}")
    
    uvicorn.run(
        "app:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )
