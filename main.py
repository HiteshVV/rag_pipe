"""
Main RAG Pipeline Implementation
Orchestrates the complete retrieval-augmented generation process
"""

import pandas as pd
from typing import Dict, List, Any, Tuple
from loguru import logger
from datetime import datetime
import json
import time

from rag_components import extract_time_filters, extract_participant_filters, build_chroma_filter
from model_setup import generate_mistral_response
from chroma_setup import query_chromadb
from timing_utils import PipelineTimer

def validate_query(query: str) -> Dict[str, Any]:
    """
    Validate if the query is meaningful and related to CDR data
    
    Args:
        query: User query string
        
    Returns:
        Dictionary with validation results
    """
    query_lower = query.strip().lower()
    
    # Check for empty or very short queries
    if len(query_lower) < 2:
        return {
            'is_valid': False,
            'reason': 'empty_query',
            'message': 'Please provide a query about call data records (CDR).'
        }
    
    # Check for generic greetings or non-CDR related queries
    invalid_patterns = [
        # Greetings
        r'^(hi|hello|hey|good morning|good afternoon|good evening)$',
        # General conversation
        r'^(thanks|thank you|bye|goodbye|ok|okay)$',
        # Random words
        r'^(test|testing|abc|xyz|123)$',
        # Single characters or numbers
        r'^[a-z]{1,2}$',
        r'^\d{1,2}$'
    ]
    
    import re
    for pattern in invalid_patterns:
        if re.match(pattern, query_lower):
            return {
                'is_valid': False,
                'reason': 'irrelevant_query',
                'message': 'This query is not related to CDR (Call Detail Records) analysis. Please ask about calls, call patterns, phone numbers, users, or call statistics.'
            }
    
    # Check for non-CDR topics that should be rejected
    non_cdr_topics = [
        'weather', 'climate', 'temperature', 'rain', 'snow',
        'food', 'recipe', 'cooking', 'restaurant', 'meal',
        'movie', 'film', 'music', 'song', 'book', 'news',
        'sports', 'football', 'basketball', 'soccer', 'game',
        'travel', 'vacation', 'hotel', 'flight', 'car',
        'health', 'doctor', 'medicine', 'hospital', 'sick',
        'school', 'education', 'homework', 'university',
        'shopping', 'buy', 'sell', 'price', 'store',
        'color', 'animal', 'dog', 'cat', 'bird'
    ]
    
    for topic in non_cdr_topics:
        if topic in query_lower and not any(cdr_word in query_lower for cdr_word in ['call', 'phone', 'number', 'record', 'cdr']):
            return {
                'is_valid': False,
                'reason': 'non_cdr_topic',
                'message': f'This query appears to be about {topic}, which is not related to CDR (Call Detail Records). Please ask about calls, phone numbers, users, or call statistics.'
            }
    
    # Check for CDR-related keywords (positive validation)
    cdr_keywords = [
        'call', 'calls', 'phone', 'number', 'extension', 'user', 'duration', 
        'time', 'date', 'incoming', 'outgoing', 'record', 'cdr', 'communication',
        'participant', 'address', 'platform', 'audio', 'recording', 'channel',
        'many', 'count', 'total', 'show', 'find', 'list', 'get', 'between',
        'from', 'to', 'when', 'how', 'what', 'who', 'which', 'where'
    ]
    
    # Check if query contains any CDR-related keywords
    has_cdr_keywords = any(keyword in query_lower for keyword in cdr_keywords)
    
    # Check for month/date patterns
    has_date_info = any(month in query_lower for month in [
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        '2024', '2025', 'today', 'yesterday', 'hour', 'day', 'week', 'month', 'year'
    ])
    
    # Check for number patterns (phone numbers, extensions)
    has_numbers = re.search(r'\b\d{3,4}\b', query_lower) is not None
    
    if not (has_cdr_keywords or has_date_info or has_numbers):
        return {
            'is_valid': False,
            'reason': 'no_cdr_context',
            'message': 'Your query doesn\'t seem to be related to CDR (Call Detail Records). Please ask about calls, phone numbers, users, call duration, dates, or call statistics.'
        }
    
    return {
        'is_valid': True,
        'reason': 'valid_query',
        'message': 'Query is valid for CDR analysis.'
    }

def analyze_query_intent(query: str) -> Dict[str, Any]:
    """
    Analyze user query to determine intent and required analysis type
    
    Args:
        query: User query string
        
    Returns:
        Dictionary with query analysis results
    """
    query_lower = query.lower()
    
    intent = {
        'type': 'general',
        'requires_stats': False,
        'requires_chart': False,
        'time_based': False,
        'participant_based': False,
        'pattern_analysis': False,
        'is_count_query': False
    }
    
    # Determine query type
    if any(word in query_lower for word in ['how many', 'count', 'total', 'number of']):
        intent['type'] = 'count'
        intent['requires_stats'] = True
        intent['is_count_query'] = True
        
    # Check for comparison queries using flexible patterns
    elif any(comparison_word in query_lower for comparison_word in ['more', 'less', 'fewer', 'greater', 'higher', 'lower', 'compare', 'versus', 'vs', 'between']):
        # Check if it's comparing call directions or types
        if any(direction in query_lower for direction in ['incoming', 'outgoing', 'inbound', 'outbound']) or \
           any(type_word in query_lower for type_word in ['calls', 'call']):
            intent['type'] = 'count'
            intent['requires_stats'] = True
            intent['is_count_query'] = True
        
    elif any(word in query_lower for word in ['pattern', 'trend', 'behavior', 'analysis']):
        intent['type'] = 'pattern'
        intent['pattern_analysis'] = True
        intent['requires_chart'] = True
        
    elif any(word in query_lower for word in ['when', 'time', 'hour', 'day', 'duration', 'breakdown', 'by month', 'by day', 'by year', 'monthly', 'daily', 'yearly', 'hourly']):
        intent['type'] = 'temporal'
        intent['time_based'] = True
        intent['requires_chart'] = True
        
    elif any(word in query_lower for word in ['who', 'between', 'user', 'participant']):
        intent['type'] = 'participant'
        intent['participant_based'] = True
        
    elif any(word in query_lower for word in ['show', 'list', 'find', 'get']):
        intent['type'] = 'retrieval'
        
    # Check for chart/visualization keywords and breakdown patterns
    if any(word in query_lower for word in ['chart', 'graph', 'plot', 'visualize', 'show me', 'breakdown', 'by month', 'by day', 'by year', 'by hour', 'monthly', 'daily', 'yearly', 'hourly']):
        intent['requires_chart'] = True
    
    logger.info(f"Query intent analysis: {intent}")
    return intent

def generate_analytics_data(df: pd.DataFrame, query: str, intent: Dict) -> Dict[str, Any]:
    """
    Generate analytics data based on query intent and dataframe
    
    Args:
        df: DataFrame with CDR records
        query: Original user query
        intent: Query intent analysis
        
    Returns:
        Dictionary with analytics data including potential charts
    """
    analytics = {}
    
    try:
        if df.empty:
            return {"error": "No data available for analysis"}
        
        query_lower = query.lower()
        
        # Time-based analytics
        if intent['time_based'] or 'hour' in query_lower:
            if 'start_time' in df.columns:
                # Convert timestamps and extract hour
                df_temp = df.copy()
                df_temp['start_time'] = pd.to_datetime(df_temp['start_time'], errors='coerce')
                df_temp['hour'] = df_temp['start_time'].dt.hour
                
                hourly_counts = df_temp.groupby('hour').size().to_dict()
                
                analytics['hourly_distribution'] = {
                    'type': 'bar',
                    'title': 'Calls by Hour of Day',
                    'labels': [f"{h}:00" for h in range(24)],
                    'values': [hourly_counts.get(h, 0) for h in range(24)],
                    'xLabel': 'Hour of Day',
                    'yLabel': 'Number of Calls'
                }
        
        # Participant analytics
        if intent['participant_based'] or any(word in query_lower for word in ['participant', 'user', 'extension']):
            if 'source_address' in df.columns and 'destination_address' in df.columns:
                # Most active participants
                source_counts = df['source_address'].value_counts().head(10)
                dest_counts = df['destination_address'].value_counts().head(10)
                
                # Combined participant activity
                all_participants = pd.concat([source_counts, dest_counts]).groupby(level=0).sum().sort_values(ascending=False).head(5)
                
                analytics['participant_activity'] = {
                    'type': 'bar',
                    'title': 'Most Active Participants',
                    'labels': list(all_participants.index),
                    'values': list(all_participants.values),
                    'xLabel': 'Participant',
                    'yLabel': 'Total Calls'
                }
        
        # Direction analytics
        if 'direction' in df.columns:
            direction_counts = df['direction'].value_counts().to_dict()
            
            analytics['call_direction'] = {
                'type': 'pie',
                'title': 'Incoming vs Outgoing Calls',
                'labels': list(direction_counts.keys()),
                'values': list(direction_counts.values)
            }
        
        # Duration analytics
        if 'duration_seconds' in df.columns:
            duration_stats = {
                'total_duration': df['duration_seconds'].sum(),
                'avg_duration': df['duration_seconds'].mean(),
                'max_duration': df['duration_seconds'].max(),
                'min_duration': df['duration_seconds'].min()
            }
            
            analytics['duration_stats'] = duration_stats
            
            # Duration distribution
            df_temp = df.copy()
            df_temp['duration_category'] = pd.cut(
                df_temp['duration_seconds'], 
                bins=[0, 10, 30, 60, 300, float('inf')],
                labels=['<10s', '10-30s', '30s-1m', '1-5m', '>5m']
            )
            
            duration_dist = df_temp['duration_category'].value_counts().to_dict()
            
            analytics['duration_distribution'] = {
                'type': 'bar',
                'title': 'Call Duration Distribution',
                'labels': list(duration_dist.keys()),
                'values': list(duration_dist.values),
                'xLabel': 'Duration Category',
                'yLabel': 'Number of Calls'
            }
        
        # Count analytics
        if intent['requires_stats'] or intent['type'] == 'count':
            total_calls = len(df)
            unique_participants = len(set(list(df['source_address'].unique()) + list(df['destination_address'].unique())))
            
            analytics['summary_stats'] = {
                'type': 'single_value',
                'title': 'Total Calls',
                'labels': ['Total Calls'],
                'values': [total_calls],
                'additional_stats': {
                    'unique_participants': unique_participants,
                    'date_range': f"{df['start_time'].min()} to {df['start_time'].max()}" if 'start_time' in df.columns else 'N/A'
                }
            }
        
        logger.info(f"Generated analytics: {list(analytics.keys())}")
        return analytics
        
    except Exception as e:
        logger.error(f"Error generating analytics: {e}")
        return {"error": f"Analytics generation failed: {str(e)}"}

def filter_dataframe_by_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Filter dataframe based on query parameters
    
    Args:
        df: Original dataframe
        query: User query
        
    Returns:
        Filtered dataframe
    """
    filtered_df = df.copy()
    query_lower = query.lower()
    
    logger.info(f"[FILTER] Starting with {len(filtered_df)} records")
    logger.info(f"[FILTER] Query: {query}")
    
    try:
        # Time filters
        time_filters = extract_time_filters(query)
        logger.info(f"[FILTER] Time filters: {time_filters}")
        
        if time_filters:
            if 'start_time' in filtered_df.columns:
                filtered_df['start_time'] = pd.to_datetime(filtered_df['start_time'], errors='coerce')
                
                if 'date' in time_filters:
                    target_date = time_filters['date']
                    logger.info(f"[FILTER] Target date: {target_date}")
                    
                    # Convert target date to datetime for comparison
                    target_datetime = pd.to_datetime(target_date)
                    logger.info(f"[FILTER] Target datetime: {target_datetime}")
                    
                    # Show sample dates before filtering
                    logger.info(f"[FILTER] Sample dates before filtering: {filtered_df['start_time'].head(3).tolist()}")
                    
                    filtered_df = filtered_df[filtered_df['start_time'].dt.date == target_datetime.date()]
                    logger.info(f"[FILTER] After date filtering: {len(filtered_df)} records")
                
                elif 'month_year' in time_filters:
                    target_month_year = time_filters['month_year']
                    filtered_df = filtered_df[filtered_df['start_time'].dt.strftime('%Y-%m') == target_month_year]
                
                elif 'year' in time_filters:
                    target_year = int(time_filters['year'])
                    filtered_df = filtered_df[filtered_df['start_time'].dt.year == target_year]
                
                elif 'time_range' in time_filters:
                    now = datetime.now()
                    if time_filters['time_range'] == 'last_hour':
                        cutoff = now - pd.Timedelta(hours=1)
                    elif time_filters['time_range'] == 'last_day':
                        cutoff = now - pd.Timedelta(days=1)
                    elif time_filters['time_range'] == 'last_week':
                        cutoff = now - pd.Timedelta(weeks=1)
                    else:
                        cutoff = None
                    
                    if cutoff:
                        filtered_df = filtered_df[filtered_df['start_time'] >= cutoff]
        
        # Participant filters
        participant_filters = extract_participant_filters(query)
        if participant_filters:
            if 'extensions' in participant_filters:
                extensions = participant_filters['extensions']
                mask = (filtered_df['source_address'].isin(extensions)) | (filtered_df['destination_address'].isin(extensions))
                filtered_df = filtered_df[mask]
            
            if 'users' in participant_filters:
                users = participant_filters['users']
                if 'user_id' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['user_id'].isin(users)]
            
            if 'between' in participant_filters:
                participants = participant_filters['between']
                mask = ((filtered_df['source_address'].isin(participants)) & (filtered_df['destination_address'].isin(participants)))
                filtered_df = filtered_df[mask]
        
        # Direction filters - but skip for comparison queries that mention both directions
        query_has_both_directions = ('incoming' in query_lower or 'inbound' in query_lower) and \
                                   ('outgoing' in query_lower or 'outbound' in query_lower)
        
        if not query_has_both_directions:
            if 'incoming' in query_lower or 'inbound' in query_lower:
                if 'direction' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['direction'] == 'incoming']
            elif 'outgoing' in query_lower or 'outbound' in query_lower:
                if 'direction' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['direction'] == 'outgoing']
        
        # Duration filters
        if 'long' in query_lower or 'longer than' in query_lower:
            if 'duration_seconds' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['duration_seconds'] > 30]
        elif 'short' in query_lower or 'brief' in query_lower:
            if 'duration_seconds' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['duration_seconds'] < 10]
        
        logger.info(f"Filtered dataframe: {len(filtered_df)} records (from {len(df)} original)")
        return filtered_df
        
    except Exception as e:
        logger.error(f"Error filtering dataframe: {e}")
        return df

def rag_pipeline(
    query: str,
    df_all_records: pd.DataFrame,
    chroma_collection: Any,
    llm_client: Any,
    embedding_model: Any,
    metadatas: List[Dict],
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Main RAG pipeline function
    
    Args:
        query: User query
        df_all_records: DataFrame with all CDR records
        chroma_collection: ChromaDB collection
        llm_client: Ollama/Mistral client
        embedding_model: Sentence transformer model
        metadatas: List of metadata dictionaries
        max_results: Maximum number of retrieval results
        
    Returns:
        Dictionary with LLM response and analytics
    """
    logger.info(f"Starting RAG pipeline for query: {query}")
    
    # Initialize pipeline timer
    timer = PipelineTimer()
    
    try:
        # 1. Validate query first
        validation = timer.time_stage("Query Validation", validate_query, query)
        
        if not validation['is_valid']:
            logger.warning(f"Invalid query detected: {validation['reason']}")
            return {
                'llm_summary': validation['message'],
                'top_snippets': [],
                'retrieved_count': 0,
                'total_matching_date': 0,
                'explanation': f"Query validation failed: {validation['reason']}",
                'analytics': {},
                'query_valid': False,
                'validation_reason': validation['reason'],
                'timing': timer.get_timing_summary()
            }
        
        # 2. Analyze query intent
        intent = timer.time_stage("Query Intent Analysis", analyze_query_intent, query)
        
        # 3. Build ChromaDB filter
        chroma_filter = build_chroma_filter(query)
        
        # 4. Check if this is a date-based query that needs special handling
        time_filters = extract_time_filters(query)
        is_date_query = time_filters and ('date' in time_filters or 'month_year' in time_filters or 'year' in time_filters)
        
        # 5. For count queries, prioritize DataFrame analysis over semantic search
        if intent['is_count_query']:
            # For count queries, we don't need detailed document retrieval
            retrieval_results = {"documents": [[]]}
            context_docs = []
        elif is_date_query:
            # For date queries, get a diverse sample and rely on post-filtering
            try:
                # Get a large sample of documents without semantic search
                sample_results = chroma_collection.get(
                    limit=min(1000, chroma_collection.count()),
                    include=['documents']
                )
                retrieval_results = {
                    'documents': [sample_results['documents']] if sample_results['documents'] else [[]]
                }
                logger.info(f"Retrieved {len(sample_results['documents'])} documents for date filtering")
            except Exception as e:
                logger.error(f"Error getting sample documents: {e}")
                retrieval_results = {"documents": [[]]}
        else:
            retrieval_results = timer.time_stage("Vector Retrieval", query_chromadb,
                collection=chroma_collection,
                query=query,
                embedding_model=embedding_model,
                n_results=max_results,
                where_filter=chroma_filter
            )
        
        # 6. Filter dataframe for analytics
        filtered_df = timer.time_stage("DataFrame Filtering", filter_dataframe_by_query, df_all_records, query)
        
        # 7. Generate analytics data
        analytics = timer.time_stage("Analytics Generation", generate_analytics_data, filtered_df, query, intent)
        
        # 8. Build context from retrieved documents
        if intent['is_count_query']:
            # For count queries, build context from DataFrame statistics
            total_calls = len(filtered_df)
            
            # Build detailed count context
            context_parts = [f"Query: {query}"]
            context_parts.append(f"Total matching calls: {total_calls}")
            
            # Check if this is a monthly breakdown query
            if any(word in query.lower() for word in ['by month', 'monthly', 'month']):
                # Add actual monthly breakdown data
                if not filtered_df.empty and 'start_time' in filtered_df.columns:
                    df_temp = filtered_df.copy()
                    df_temp['start_time'] = pd.to_datetime(df_temp['start_time'], errors='coerce')
                    monthly_counts = df_temp.groupby(df_temp['start_time'].dt.to_period('M')).size()
                    
                    context_parts.append("\nActual Monthly Breakdown:")
                    for month_period, count in monthly_counts.items():
                        month_name = month_period.strftime('%B %Y')  # e.g., "January 2025"
                        context_parts.append(f"- {month_name}: {count} calls")
                    
                    context_parts.append(f"\nTotal calls across all months: {monthly_counts.sum()}")
            
            elif any(word in query.lower() for word in ['by day', 'daily', 'day']):
                # Add daily breakdown if requested
                if not filtered_df.empty and 'start_time' in filtered_df.columns:
                    df_temp = filtered_df.copy()
                    df_temp['start_time'] = pd.to_datetime(df_temp['start_time'], errors='coerce')
                    daily_counts = df_temp.groupby(df_temp['start_time'].dt.date).size()
                    
                    context_parts.append(f"\nDaily Breakdown (showing up to 10 days):")
                    for date, count in daily_counts.head(10).items():
                        context_parts.append(f"- {date}: {count} calls")
                    
                    if len(daily_counts) > 10:
                        context_parts.append(f"... and {len(daily_counts) - 10} more days")
            
            if not filtered_df.empty:
                # Add breakdown by direction if relevant
                if 'direction' in filtered_df.columns:
                    direction_counts = filtered_df['direction'].value_counts()
                    context_parts.append("\nCall Direction Breakdown:")
                    for direction, count in direction_counts.items():
                        context_parts.append(f"- {direction.capitalize()} calls: {count}")
                
                # Add time period info
                if 'start_time' in filtered_df.columns:
                    start_date = filtered_df['start_time'].min()
                    end_date = filtered_df['start_time'].max()
                    context_parts.append(f"\nDate Range: {start_date} to {end_date}")
                
                # Add participant info
                if 'source_address' in filtered_df.columns and 'destination_address' in filtered_df.columns:
                    unique_sources = filtered_df['source_address'].nunique()
                    unique_destinations = filtered_df['destination_address'].nunique()
                    context_parts.append(f"- Unique calling numbers: {unique_sources}")
                    context_parts.append(f"- Unique called numbers: {unique_destinations}")
            
            context = "\n".join(context_parts)
            context_docs = []
            
        elif intent['time_based'] or 'hour' in query.lower():
            # For hourly trend queries, provide summarized hourly breakdown
            context_parts = [f"Query: {query}"]
            context_parts.append(f"Total calls analyzed: {len(filtered_df)}")
            
            if not filtered_df.empty and 'start_time' in filtered_df.columns:
                # Create simplified hourly breakdown
                df_temp = filtered_df.copy()
                df_temp['start_time'] = pd.to_datetime(df_temp['start_time'], errors='coerce')
                df_temp['hour'] = df_temp['start_time'].dt.hour
                
                hourly_counts = df_temp.groupby('hour').size()
                
                context_parts.append("\nHourly Call Distribution:")
                context_parts.append(f"Peak hours: {hourly_counts.nlargest(3).to_dict()}")
                context_parts.append(f"Low activity hours: {hourly_counts.nsmallest(3).to_dict()}")
                context_parts.append(f"Total hours with activity: {len(hourly_counts)}")
                
                # Add direction summary
                if 'direction' in df_temp.columns:
                    direction_by_hour = df_temp.groupby(['hour', 'direction']).size().unstack(fill_value=0)
                    context_parts.append(f"Direction distribution varies by hour: {direction_by_hour.sum().to_dict()}")
            
            context = "\n".join(context_parts)
            context_docs = []
            
        else:
            context_docs = []
            if retrieval_results and 'documents' in retrieval_results and retrieval_results['documents']:
                all_docs = retrieval_results['documents'][0]
                
                # Post-filter documents for date queries
                if is_date_query and time_filters:
                    filtered_docs = []
                    
                    if 'date' in time_filters:
                        target_date = time_filters['date']
                        logger.info(f"Post-filtering documents for date: {target_date}")
                        for doc in all_docs:
                            if target_date in doc or target_date.replace('-', '/') in doc or target_date.replace('-', ' ') in doc:
                                filtered_docs.append(doc)
                    
                    elif 'month_year' in time_filters:
                        target_month_year = time_filters['month_year']
                        logger.info(f"Post-filtering documents for month-year: {target_month_year}")
                        for doc in all_docs:
                            if target_month_year in doc:
                                filtered_docs.append(doc)
                    
                    elif 'year' in time_filters:
                        target_year = time_filters['year']
                        logger.info(f"Post-filtering documents for year: {target_year}")
                        for doc in all_docs:
                            if target_year in doc:
                                filtered_docs.append(doc)
                    
                    context_docs = filtered_docs[:max_results]
                    logger.info(f"Found {len(filtered_docs)} documents after date filtering (showing {len(context_docs)})")
                    
                else:
                    context_docs = all_docs[:max_results]
        
            context = "\n\n".join(context_docs) if context_docs else "No relevant CDR records found."        # 8. Add analytics summary to context
        if analytics and 'summary_stats' in analytics:
            stats = analytics['summary_stats']
            context += f"\n\nSummary Statistics:\n"
            context += f"- Total calls in dataset: {stats['values'][0]}\n"
            if 'additional_stats' in stats:
                context += f"- Unique participants: {stats['additional_stats'].get('unique_participants', 'N/A')}\n"
                context += f"- Date range: {stats['additional_stats'].get('date_range', 'N/A')}\n"
        
        # For count queries, always add count information
        if intent['is_count_query'] or is_date_query:
            filtered_count = len(filtered_df)
            context += f"\n\nQuery Results:\n"
            context += f"- Calls matching your query: {filtered_count}\n"
            
            if time_filters:
                if 'date' in time_filters:
                    context += f"- Query date: {time_filters['date']}\n"
                elif 'month_year' in time_filters:
                    context += f"- Query month: {time_filters['month_year']}\n"
                elif 'year' in time_filters:
                    context += f"- Query year: {time_filters['year']}\n"
        
        # 9. Generate LLM response
        llm_response = timer.time_stage("LLM Generation", generate_mistral_response,
            client=llm_client,
            query=query,
            context=context
        )
        
        # 10. Prepare response
        total_found_for_date = 0
        if is_date_query and time_filters and 'date' in time_filters:
            # Count total documents that match the date
            target_date = time_filters['date']
            if retrieval_results and 'documents' in retrieval_results and retrieval_results['documents']:
                all_docs = retrieval_results['documents'][0]
                for doc in all_docs:
                    if target_date in doc or target_date.replace('-', '/') in doc or target_date.replace('-', ' ') in doc:
                        total_found_for_date += 1
        
        # Remove the explanation_text line from the response
        response = {
            "llm_summary": llm_response,
            "top_snippets": context_docs[:3] if context_docs else [],
            "retrieved_count": len(context_docs),
            "total_matching_date": total_found_for_date if is_date_query else None,
            # "explanation": explanation_text,  # Removed as per user request
            "query_intent": intent
        }
        
        # Log timing summary
        timer.log_summary()
        
        # 10. Add chart data if analytics were generated
        if analytics:
            # Find the most relevant chart for the query
            if intent['time_based'] and 'hourly_distribution' in analytics:
                response['graph_data'] = analytics['hourly_distribution']
            elif intent['participant_based'] and 'participant_activity' in analytics:
                response['graph_data'] = analytics['participant_activity']
            elif intent['type'] == 'count' and 'summary_stats' in analytics:
                response['graph_data'] = analytics['summary_stats']
            elif 'call_direction' in analytics:
                response['graph_data'] = analytics['call_direction']
            elif 'duration_distribution' in analytics:
                response['graph_data'] = analytics['duration_distribution']
        
        logger.info("RAG pipeline completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {e}")
        return {
            "llm_summary": f"Error processing query: {str(e)}",
            "top_snippets": [],
            "explanation": "An error occurred while processing your query.",
            "error": str(e)
        }
