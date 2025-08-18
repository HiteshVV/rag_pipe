"""
Model Setup and Initialization
Handles embedding models and LLM pipeline setup
"""

from sentence_transformers import SentenceTransformer
import ollama
from loguru import logger
from typing import Tuple, Any

from config.settings import EMBEDDING_MODEL_NAME, OLLAMA_BASE_URL, MISTRAL_MODEL_NAME

def initialize_embedding_model(model_name: str = None) -> SentenceTransformer:
    """
    Initialize the sentence transformer embedding model
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        Initialized SentenceTransformer model
    """
    if model_name is None:
        model_name = EMBEDDING_MODEL_NAME
    
    logger.info(f"Loading embedding model: {model_name}")
    
    try:
        model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        raise

def initialize_ollama_client(base_url: str = None) -> Any:
    """
    Initialize Ollama client for Mistral LLM
    
    Args:
        base_url: Ollama server base URL
        
    Returns:
        Ollama client instance
    """
    if base_url is None:
        base_url = OLLAMA_BASE_URL
    
    logger.info(f"Initializing Ollama client at {base_url}")
    
    try:
        # Set the Ollama client with custom base URL if needed
        client = ollama
        
        # Test connection by listing models
        models = client.list()
        
        # Handle different response formats for models
        model_names = []
        if hasattr(models, 'models') and models.models:
            try:
                # Try pydantic-style access first
                model_names = [getattr(m, 'model', getattr(m, 'name', str(m))) for m in models.models]
            except AttributeError:
                # Fallback to dict-style access
                try:
                    model_names = [m.get('name', m.get('model', str(m))) if isinstance(m, dict) else str(m) for m in models.models]
                except:
                    model_names = [str(m) for m in models.models]
        elif isinstance(models, dict) and 'models' in models:
            model_names = [m.get('name', m.get('model', 'unknown')) for m in models['models'] if isinstance(m, dict)]
        
        logger.info(f"Connected to Ollama. Available models: {model_names}")
        
        # Check if mistral model is available
        if not any('mistral' in str(name).lower() for name in model_names):
            logger.warning("Mistral model not found. Please run: ollama pull mistral")
            
        return client
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {e}")
        logger.error("Make sure Ollama is running and Mistral model is installed")
        logger.error("Run: ollama pull mistral")
        raise

def initialize_models() -> Tuple[SentenceTransformer, Any, str, str]:
    """
    Initialize all models needed for the RAG pipeline
    
    Returns:
        Tuple of (embedding_model, llm_client, llm_model_name, embedding_model_name)
    """
    logger.info("Initializing RAG pipeline models...")
    
    # Initialize embedding model
    embedding_model = initialize_embedding_model()
    
    # Initialize LLM client
    llm_client = initialize_ollama_client()
    
    logger.info("All models initialized successfully")
    
    return embedding_model, llm_client, MISTRAL_MODEL_NAME, EMBEDDING_MODEL_NAME

def test_mistral_connection(client: Any, model_name: str = None) -> bool:
    """
    Test connection to Mistral model
    
    Args:
        client: Ollama client
        model_name: Mistral model name
        
    Returns:
        True if connection successful
    """
    if model_name is None:
        model_name = MISTRAL_MODEL_NAME
    
    try:
        logger.info(f"Testing Mistral model: {model_name}")
        
        response = client.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": "Hello, respond with just 'OK' to confirm you're working."
                }
            ]
        )
        
        if response and 'message' in response:
            logger.info("Mistral model connection successful")
            return True
        else:
            logger.error("Unexpected response from Mistral")
            return False
            
    except Exception as e:
        logger.error(f"Error testing Mistral connection: {e}")
        return False

def generate_mistral_response(
    client: Any, 
    query: str, 
    context: str, 
    model_name: str = None
) -> str:
    """
    Generate response using Mistral model
    
    Args:
        client: Ollama client
        query: User query
        context: Retrieved context from ChromaDB
        model_name: Mistral model name
        
    Returns:
        Generated response
    """
    if model_name is None:
        model_name = MISTRAL_MODEL_NAME
    
    # Create system prompt for CDR analysis
    system_prompt = """You are an expert telecommunications data analyst specializing in Call Detail Records (CDR). 

Your role is to:
1. Analyze CDR data patterns and provide insights
2. Answer questions about call records, participants, and communication patterns
3. Identify trends, anomalies, and important metrics
4. Provide clear, concise explanations of telecommunications data

When responding:
- Be specific and data-driven
- Use proper telecommunications terminology
- Highlight important patterns or anomalies
- Provide context for your findings
- If asked about specific calls, include relevant details like time, duration, participants, and direction

Focus on being helpful and accurate in your analysis of the CDR data."""

    # Create user prompt with context
    user_prompt = f"""Based on the following CDR (Call Detail Records) data, please answer the user's question.

CDR Context:
{context}

User Question: {query}

Please provide a comprehensive analysis based on the available data."""

    try:
        logger.info(f"Generating Mistral response for query: {query[:100]}...")
        
        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Handle different response formats
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            generated_text = response.message.content
        elif isinstance(response, dict) and 'message' in response and 'content' in response['message']:
            generated_text = response['message']['content']
        elif isinstance(response, dict) and 'content' in response:
            generated_text = response['content']
        elif isinstance(response, str):
            generated_text = response
        else:
            logger.error(f"Unexpected response structure: {type(response)}")
            return "Error: Unable to parse response from Mistral model."
            
        if generated_text:
            logger.info("Mistral response generated successfully")
            return generated_text
        else:
            logger.error("Empty response from Mistral")
            return "Error: Empty response from Mistral model."
            
    except Exception as e:
        logger.error(f"Error generating Mistral response: {e}")
        return f"Error: {str(e)}"
