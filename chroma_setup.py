"""
ChromaDB Setup and Management
Handles vector database initialization and operations
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from loguru import logger
from typing import List, Dict, Any, Optional
import os

from config.settings import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME

def initialize_chromadb(
    documents: List[str],
    metadatas: List[Dict],
    ids: List[str],
    embedding_model: SentenceTransformer,
    embedding_model_name: str,
    persist_directory: str = None
) -> Any:
    """
    Initialize ChromaDB collection with CDR documents
    
    Args:
        documents: List of document strings
        metadatas: List of metadata dictionaries
        ids: List of document IDs
        embedding_model: SentenceTransformer model for embeddings
        embedding_model_name: Name of the embedding model
        persist_directory: Directory to persist ChromaDB
        
    Returns:
        ChromaDB collection object
    """
    if persist_directory is None:
        persist_directory = str(CHROMA_DB_DIR)
    
    logger.info(f"Initializing ChromaDB at {persist_directory}")
    
    # Ensure directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    try:
        # Initialize ChromaDB client with persistence
        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        collection_name = f"{CHROMA_COLLECTION_NAME}_{embedding_model_name.replace('/', '_')}"
        
        try:
            # Try to get existing collection
            collection = client.get_collection(name=collection_name)
            existing_count = collection.count()
            
            if existing_count > 0:
                logger.info(f"Found existing collection '{collection_name}' with {existing_count} documents")
                
                # Check if we need to add new documents
                if existing_count < len(documents):
                    logger.info(f"Adding {len(documents) - existing_count} new documents")
                    add_documents_to_collection(
                        collection, 
                        documents[existing_count:], 
                        metadatas[existing_count:], 
                        ids[existing_count:], 
                        embedding_model
                    )
                else:
                    logger.info("Collection is up to date")
                
                return collection
                
        except Exception:
            # Collection doesn't exist, create new one
            logger.info(f"Creating new collection '{collection_name}'")
            
        # Create new collection
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add all documents
        add_documents_to_collection(collection, documents, metadatas, ids, embedding_model)
        
        logger.info(f"ChromaDB collection '{collection_name}' initialized successfully")
        return collection
        
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {e}")
        raise

def add_documents_to_collection(
    collection: Any,
    documents: List[str],
    metadatas: List[Dict],
    ids: List[str],
    embedding_model: SentenceTransformer,
    batch_size: int = 100
) -> None:
    """
    Add documents to ChromaDB collection in batches
    
    Args:
        collection: ChromaDB collection
        documents: List of document strings
        metadatas: List of metadata dictionaries
        ids: List of document IDs
        embedding_model: SentenceTransformer model
        batch_size: Number of documents to process at once
    """
    logger.info(f"Adding {len(documents)} documents to collection in batches of {batch_size}")
    
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        batch_docs = documents[i:batch_end]
        batch_metas = metadatas[i:batch_end]
        batch_ids = ids[i:batch_end]
        
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        try:
            # Generate embeddings for batch
            embeddings = embedding_model.encode(batch_docs)
            
            # Convert to list of lists (ChromaDB format)
            embeddings_list = embeddings.tolist()
            
            # Add to collection
            collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids,
                embeddings=embeddings_list
            )
            
            logger.info(f"Successfully added batch {i//batch_size + 1}")
            
        except Exception as e:
            logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
            continue
    
    final_count = collection.count()
    logger.info(f"Collection now contains {final_count} documents")

def query_chromadb(
    collection: Any,
    query: str,
    embedding_model: SentenceTransformer,
    n_results: int = 10,
    where_filter: Dict = None
) -> Dict:
    """
    Query ChromaDB collection with semantic search
    
    Args:
        collection: ChromaDB collection
        query: Query string
        embedding_model: SentenceTransformer model
        n_results: Number of results to return
        where_filter: Metadata filter dictionary
        
    Returns:
        Query results dictionary
    """
    logger.info(f"Querying ChromaDB: '{query[:100]}...' (top {n_results} results)")
    
    try:
        # Check if this is a date-based query that needs post-processing
        from rag_components import extract_time_filters
        time_filters = extract_time_filters(query)
        needs_date_filtering = time_filters and 'date' in time_filters
        
        # If we need date filtering, get more results to filter later
        search_results = n_results * 3 if needs_date_filtering else n_results
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query])
        query_embedding_list = query_embedding.tolist()
        
        # Build query parameters
        query_params = {
            "query_embeddings": query_embedding_list,
            "n_results": search_results,
            "include": ["documents", "metadatas", "distances"]
        }
        
        if where_filter:
            query_params["where"] = where_filter
            logger.info(f"Applying filter: {where_filter}")
        
        # Execute query
        results = collection.query(**query_params)
        
        # Post-process for date filtering
        if needs_date_filtering and results and 'documents' in results and results['documents']:
            target_date = time_filters['date']
            logger.info(f"Post-filtering results for date: {target_date}")
            
            filtered_docs = []
            filtered_metas = []
            filtered_distances = []
            
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            for doc, meta, dist in zip(documents, metadatas, distances):
                start_time = meta.get('start_time', '')
                if start_time.startswith(target_date):
                    filtered_docs.append(doc)
                    filtered_metas.append(meta)
                    filtered_distances.append(dist)
                    
                    # Stop when we have enough results
                    if len(filtered_docs) >= n_results:
                        break
            
            logger.info(f"Date filtering: {len(filtered_docs)} results after filtering from {len(documents)} original results")
            
            # Return filtered results
            results = {
                "documents": [filtered_docs],
                "metadatas": [filtered_metas], 
                "distances": [filtered_distances]
            }
        
        # Log results summary
        if results and 'documents' in results and results['documents']:
            num_results = len(results['documents'][0])
            logger.info(f"Retrieved {num_results} relevant documents")
        else:
            logger.warning("No results found")
        
        return results
        
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

def get_collection_stats(collection: Any) -> Dict:
    """
    Get statistics about the ChromaDB collection
    
    Args:
        collection: ChromaDB collection
        
    Returns:
        Dictionary with collection statistics
    """
    try:
        total_docs = collection.count()
        
        # Get sample of documents to analyze metadata
        sample_results = collection.query(
            query_embeddings=[[0.0] * 384],  # Dummy embedding
            n_results=min(100, total_docs)
        )
        
        stats = {
            "total_documents": total_docs,
            "collection_name": collection.name
        }
        
        if sample_results and 'metadatas' in sample_results and sample_results['metadatas']:
            metadatas = sample_results['metadatas'][0]
            
            # Analyze metadata fields
            directions = [m.get('direction', '') for m in metadatas]
            users = [m.get('user_id', '') for m in metadatas]
            platforms = [m.get('platform', '') for m in metadatas]
            
            stats.update({
                "unique_directions": len(set(filter(None, directions))),
                "unique_users": len(set(filter(None, users))),
                "unique_platforms": len(set(filter(None, platforms))),
                "sample_directions": list(set(filter(None, directions)))[:5],
                "sample_users": list(set(filter(None, users)))[:5],
                "sample_platforms": list(set(filter(None, platforms)))[:5]
            })
        
        logger.info(f"Collection stats: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        return {"total_documents": 0, "error": str(e)}

def reset_chromadb(persist_directory: str = None) -> bool:
    """
    Reset/clear ChromaDB database
    
    Args:
        persist_directory: Directory containing ChromaDB
        
    Returns:
        True if successful
    """
    if persist_directory is None:
        persist_directory = str(CHROMA_DB_DIR)
    
    logger.warning(f"Resetting ChromaDB at {persist_directory}")
    
    try:
        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=True)
        )
        
        client.reset()
        logger.info("ChromaDB reset successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error resetting ChromaDB: {e}")
        return False
