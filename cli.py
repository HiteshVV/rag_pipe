"""
CLI Tool for CDR RAG Pipeline
Command-line interface for testing and running queries
"""

import argparse
import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from main import rag_pipeline
from rag_components import load_es_data, prepare_documents_for_chroma_and_df
from model_setup import initialize_models, test_mistral_connection
from chroma_setup import initialize_chromadb, get_collection_stats, reset_chromadb
from config.settings import ES_DATA_FILE

def setup_pipeline():
    """Initialize the RAG pipeline components"""
    print("🚀 Initializing CDR RAG Pipeline...")
    
    # Load data
    print("📊 Loading CDR data...")
    es_raw_hits = load_es_data(str(ES_DATA_FILE))
    
    if not es_raw_hits:
        print(f"❌ No CDR data found in {ES_DATA_FILE}")
        return None, None, None, None, None, None, None
    
    print(f"✅ Loaded {len(es_raw_hits)} CDR records")
    
    # Prepare documents
    print("📝 Preparing documents...")
    documents, metadatas, ids, df_all_records = prepare_documents_for_chroma_and_df(es_raw_hits)
    print(f"✅ Prepared {len(documents)} documents")
    
    # Initialize models
    print("🤖 Initializing models...")
    embedding_model, llm_client, model_name, embedding_model_name = initialize_models()
    
    # Test Mistral connection
    print("🧪 Testing Mistral connection...")
    if test_mistral_connection(llm_client, model_name):
        print("✅ Mistral connection successful")
    else:
        print("⚠️  Mistral connection failed - queries will not work properly")
    
    # Initialize ChromaDB
    print("🔍 Initializing ChromaDB...")
    chroma_collection = initialize_chromadb(
        documents, metadatas, ids, embedding_model, embedding_model_name
    )
    
    print("✅ RAG pipeline ready!")
    return documents, metadatas, ids, df_all_records, embedding_model, llm_client, chroma_collection

def run_query(query, components):
    """Run a single query through the pipeline"""
    documents, metadatas, ids, df_all_records, embedding_model, llm_client, chroma_collection = components
    
    print(f"\n🔍 Query: {query}")
    print("⏳ Processing...")
    
    start_time = datetime.now()
    
    try:
        result = rag_pipeline(
            query=query,
            df_all_records=df_all_records,
            chroma_collection=chroma_collection,
            llm_client=llm_client,
            embedding_model=embedding_model,
            metadatas=metadatas
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n📋 Results (took {duration:.2f}s):")
        print("=" * 50)
        
        if result.get('llm_summary'):
            print("🤖 LLM Summary:")
            print(result['llm_summary'])
            print()
        
        if result.get('top_snippets'):
            print(f"📄 Top {len(result['top_snippets'])} Relevant Snippets:")
            for i, snippet in enumerate(result['top_snippets'], 1):
                print(f"{i}. {snippet[:200]}...")
            print()
        
        if result.get('explanation'):
            print(f"💡 Explanation: {result['explanation']}")
            print()
        
        if result.get('graph_data'):
            graph = result['graph_data']
            print(f"📊 Chart Data: {graph.get('title', 'Chart')}")
            print(f"   Type: {graph.get('type', 'unknown')}")
            if graph.get('labels') and graph.get('values'):
                for label, value in zip(graph['labels'][:5], graph['values'][:5]):
                    print(f"   {label}: {value}")
            if len(graph.get('labels', [])) > 5:
                print(f"   ... and {len(graph['labels']) - 5} more")
            print()
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def interactive_mode(components):
    """Run in interactive mode"""
    print("\n🎯 Interactive Mode - Type 'quit' to exit")
    print("Example queries:")
    print("  - How many calls were made?")
    print("  - Show me calls between 5656 and 5651")
    print("  - What's the calling pattern by hour?")
    print("  - Find calls longer than 30 seconds")
    print()
    
    while True:
        try:
            query = input("💬 Enter your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            run_query(query, components)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            break

def show_stats(components):
    """Show statistics about the data and collection"""
    documents, metadatas, ids, df_all_records, embedding_model, llm_client, chroma_collection = components
    
    print("\n📊 Pipeline Statistics:")
    print("=" * 50)
    
    # DataFrame stats
    if df_all_records is not None and not df_all_records.empty:
        print(f"📋 CDR Records: {len(df_all_records)}")
        
        if 'start_time' in df_all_records.columns:
            df_temp = df_all_records.copy()
            df_temp['start_time'] = pd.to_datetime(df_temp['start_time'], errors='coerce')
            print(f"📅 Date Range: {df_temp['start_time'].min()} to {df_temp['start_time'].max()}")
        
        if 'direction' in df_all_records.columns:
            direction_counts = df_all_records['direction'].value_counts()
            print(f"📞 Call Directions: {dict(direction_counts)}")
        
        if 'duration_seconds' in df_all_records.columns:
            avg_duration = df_all_records['duration_seconds'].mean()
            total_duration = df_all_records['duration_seconds'].sum()
            print(f"⏱️  Average Duration: {avg_duration:.1f}s")
            print(f"⏱️  Total Duration: {total_duration:.1f}s ({total_duration/3600:.1f} hours)")
        
        unique_sources = df_all_records['source_address'].nunique() if 'source_address' in df_all_records.columns else 0
        unique_destinations = df_all_records['destination_address'].nunique() if 'destination_address' in df_all_records.columns else 0
        print(f"👥 Unique Participants: {max(unique_sources, unique_destinations)}")
    
    # ChromaDB stats
    if chroma_collection:
        chroma_stats = get_collection_stats(chroma_collection)
        print(f"🔍 ChromaDB Documents: {chroma_stats.get('total_documents', 0)}")
        print(f"🔍 Collection Name: {chroma_stats.get('collection_name', 'unknown')}")

def main():
    parser = argparse.ArgumentParser(description="CDR RAG Pipeline CLI")
    parser.add_argument("--query", "-q", help="Run a single query and exit")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--stats", "-s", action="store_true", help="Show statistics about the data")
    parser.add_argument("--reset-db", action="store_true", help="Reset ChromaDB database")
    parser.add_argument("--test", "-t", action="store_true", help="Run test queries")
    
    args = parser.parse_args()
    
    if args.reset_db:
        print("🗑️  Resetting ChromaDB...")
        if reset_chromadb():
            print("✅ ChromaDB reset successfully")
        else:
            print("❌ Failed to reset ChromaDB")
        return
    
    # Initialize pipeline
    components = setup_pipeline()
    if not all(components):
        print("❌ Failed to initialize pipeline")
        return
    
    if args.stats:
        show_stats(components)
        return
    
    if args.test:
        print("\n🧪 Running test queries...")
        test_queries = [
            "How many calls are in the dataset?",
            "Show me calls by hour of day",
            "What's the pattern of incoming vs outgoing calls?",
            "Find calls involving extension 5651",
            "Show me the longest calls"
        ]
        
        for query in test_queries:
            run_query(query, components)
            print("-" * 50)
        return
    
    if args.query:
        run_query(args.query, components)
        return
    
    if args.interactive:
        interactive_mode(components)
        return
    
    # Default: show help
    parser.print_help()

if __name__ == "__main__":
    import pandas as pd  # Import here to avoid issues with CLI-only usage
    main()
