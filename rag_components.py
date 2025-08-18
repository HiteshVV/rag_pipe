"""
RAG Components for CDR Data Processing
Handles data loading, document preparation, and pipeline utilities
"""

import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Any
from loguru import logger
import re

from config.settings import ES_DATA_FILE

def load_es_data(file_path: str = None) -> List[Dict]:
    """
    Load CDR data from Elasticsearch export JSON file
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of CDR record hits
    """
    if file_path is None:
        file_path = ES_DATA_FILE
    
    logger.info(f"Loading CDR data from {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        hits = data.get('hits', {}).get('hits', [])
        logger.info(f"Loaded {len(hits)} CDR records")
        return hits
    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return []

def extract_cdr_fields(record: Dict) -> Dict:
    """
    Extract relevant fields from a CDR record
    
    Args:
        record: Single CDR record from Elasticsearch
        
    Returns:
        Dictionary with extracted fields
    """
    source = record.get('_source', {})
    fields = source.get('fields', {})
    
    # Extract communication record details
    comm_record = fields.get('communicationRecord', {})
    device_info = fields.get('deviceInfo', {})
    participants = fields.get('participants', [])
    com_files = fields.get('comFiles', [])
    
    # Extract participant information
    source_participant = ''
    destination_participant = ''
    
    for participant in participants:
        if participant.get('involvement') == 'source':
            source_participant = participant.get('address') or ''
        elif participant.get('involvement') == 'destination':
            destination_participant = participant.get('address') or ''
    
    # Calculate duration from communication record
    start_time = comm_record.get('startTime', '')
    end_time = comm_record.get('endTime', '')
    duration_seconds = 0
    
    if start_time and end_time:
        try:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            duration_seconds = (end_dt - start_dt).total_seconds()
        except:
            pass
    
    # Extract audio file information
    audio_files = [f for f in com_files if f.get('fileType') == 'audio']
    has_recording = len(audio_files) > 0
    
    extracted = {
        'record_id': record.get('_id') or '',
        'com_uuid': comm_record.get('comUUID') or '',
        'start_time': start_time or '',
        'end_time': end_time or '',
        'duration_seconds': duration_seconds or 0,
        'direction': comm_record.get('direction') or '',
        'com_type': comm_record.get('comType') or '',
        'platform': (comm_record.get('platform') or {}).get('name') or '',
        'source_address': source_participant or '',
        'destination_address': destination_participant or '',
        'user_id': device_info.get('pUserID') or '',
        'channel_id': device_info.get('channelID') or '',
        'audio_interface': device_info.get('audioInterface') or '',
        'button_name': device_info.get('buttonName') or '',
        'has_recording': has_recording,
        'active_duration': comm_record.get('activeDuration') or 0,
        'universal_id': comm_record.get('universalID') or '',
        'timestamp': source.get('@timestamp') or '',
        'timestamp_end': source.get('@timestampend') or '',
        'audio_files_count': len(audio_files),
        'total_files_count': len(com_files)
    }
    
    return extracted

def create_searchable_document(cdr_fields: Dict) -> str:
    """
    Create a searchable text document from CDR fields
    
    Args:
        cdr_fields: Extracted CDR fields
        
    Returns:
        Formatted text document for embedding
    """
    # Format duration for readability
    duration = cdr_fields.get('duration_seconds', 0)
    if duration >= 60:
        duration_str = f"{int(duration // 60)}m {int(duration % 60)}s"
    else:
        duration_str = f"{int(duration)}s"
    
    # Format timestamp
    timestamp = cdr_fields.get('start_time', '')
    formatted_time = ''
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            formatted_time = timestamp
    
    # Create comprehensive searchable text
    document = f"""Call Record:
- From: {cdr_fields.get('source_address', 'Unknown')} 
- To: {cdr_fields.get('destination_address', 'Unknown')}
- Time: {formatted_time}
- Direction: {cdr_fields.get('direction', 'unknown')} call
- Duration: {duration_str}
- User: {cdr_fields.get('user_id', 'Unknown')}
- Platform: {cdr_fields.get('platform', 'Unknown')}
- Audio Interface: {cdr_fields.get('audio_interface', 'Unknown')}
- Button: {cdr_fields.get('button_name', 'Unknown')}
- Channel: {cdr_fields.get('channel_id', 'Unknown')}
- Recording Available: {'Yes' if cdr_fields.get('has_recording') else 'No'}
- Call Type: {cdr_fields.get('com_type', 'Unknown')}
- Audio Files: {cdr_fields.get('audio_files_count', 0)} files
- UUID: {cdr_fields.get('com_uuid', '')}"""
    
    return document

def prepare_documents_for_chroma_and_df(es_raw_hits: List[Dict]) -> Tuple[List[str], List[Dict], List[str], pd.DataFrame]:
    """
    Prepare CDR data for ChromaDB storage and DataFrame analysis
    
    Args:
        es_raw_hits: Raw CDR records from Elasticsearch
        
    Returns:
        Tuple of (documents, metadatas, ids, dataframe)
    """
    logger.info(f"Preparing {len(es_raw_hits)} CDR records for processing")
    
    documents = []
    metadatas = []
    ids = []
    df_records = []
    
    for i, record in enumerate(es_raw_hits):
        try:
            # Extract fields
            cdr_fields = extract_cdr_fields(record)
            
            # Create searchable document
            document = create_searchable_document(cdr_fields)
            
            # Prepare metadata for ChromaDB (handle None values)
            def safe_get(field_name, default_value):
                value = cdr_fields.get(field_name, default_value)
                return default_value if value is None else value
            
            metadata = {
                'source_address': safe_get('source_address', ''),
                'destination_address': safe_get('destination_address', ''),
                'direction': safe_get('direction', ''),
                'user_id': safe_get('user_id', ''),
                'platform': safe_get('platform', ''),
                'com_type': safe_get('com_type', ''),
                'duration_seconds': safe_get('duration_seconds', 0),
                'has_recording': safe_get('has_recording', False),
                'start_time': safe_get('start_time', ''),
                'channel_id': safe_get('channel_id', ''),
                'audio_interface': safe_get('audio_interface', '')
            }
            
            # Create unique ID
            record_id = cdr_fields.get('record_id', f'record_{i}')
            
            documents.append(document)
            metadatas.append(metadata)
            ids.append(record_id)
            df_records.append(cdr_fields)
            
        except Exception as e:
            logger.warning(f"Error processing record {i}: {e}")
            continue
    
    # Create DataFrame for analysis
    df_all_records = pd.DataFrame(df_records)
    
    logger.info(f"Successfully prepared {len(documents)} documents")
    return documents, metadatas, ids, df_all_records

def extract_time_filters(query: str) -> Dict[str, Any]:
    """
    Extract time-based filters from natural language query
    
    Args:
        query: User query string
        
    Returns:
        Dictionary with time filters
    """
    filters = {}
    query_lower = query.lower()
    
    # Look for time-related keywords
    if 'today' in query_lower:
        today = datetime.now().strftime('%Y-%m-%d')
        filters['date'] = today
    elif 'yesterday' in query_lower:
        yesterday = (datetime.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        filters['date'] = yesterday
    elif 'last hour' in query_lower:
        filters['time_range'] = 'last_hour'
    elif 'last day' in query_lower or 'last 24 hours' in query_lower:
        filters['time_range'] = 'last_day'
    elif 'last week' in query_lower:
        filters['time_range'] = 'last_week'
    
    # Look for specific date patterns
    # Pattern 1: YYYY-MM-DD format
    date_pattern_1 = r'(\d{4}-\d{1,2}-\d{1,2})'
    date_matches_1 = re.findall(date_pattern_1, query)
    if date_matches_1:
        # Normalize the date format to YYYY-MM-DD
        try:
            parsed_date = datetime.strptime(date_matches_1[0], '%Y-%m-%d')
            filters['date'] = parsed_date.strftime('%Y-%m-%d')
        except ValueError:
            pass
    
    # Pattern 2: Month Day, Year format (e.g., "June 12, 2025" or "June 12th 2025" or "jan 15, 2025")
    month_pattern = r'(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})'
    month_matches = re.findall(month_pattern, query_lower)
    if month_matches:
        month_name, day, year = month_matches[0]
        month_map = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3, 
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 
            'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        try:
            month_num = month_map[month_name]
            parsed_date = datetime(int(year), month_num, int(day))
            filters['date'] = parsed_date.strftime('%Y-%m-%d')
        except (ValueError, KeyError):
            pass
    
    # Pattern 3: MM/DD/YYYY format
    date_pattern_3 = r'(\d{1,2})/(\d{1,2})/(\d{4})'
    date_matches_3 = re.findall(date_pattern_3, query)
    if date_matches_3:
        month, day, year = date_matches_3[0]
        try:
            parsed_date = datetime(int(year), int(month), int(day))
            filters['date'] = parsed_date.strftime('%Y-%m-%d')
        except ValueError:
            pass
    
    # Pattern 4: Month Year format (e.g., "June 2025", "January 2025", "jan 2025", "feb 2025")
    month_year_pattern = r'(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)\s+(\d{4})'
    month_year_matches = re.findall(month_year_pattern, query_lower)
    if month_year_matches:
        month_name, year = month_year_matches[0]
        month_map = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3, 
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 
            'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        try:
            month_num = month_map[month_name]
            filters['month_year'] = f"{year}-{month_num:02d}"
        except (ValueError, KeyError):
            pass
    
    # Pattern 5: Year only (e.g., "2025")
    year_pattern = r'\b(20\d{2})\b'
    year_matches = re.findall(year_pattern, query)
    if year_matches and 'month_year' not in filters and 'date' not in filters:
        filters['year'] = year_matches[0]
    
    # Look for specific time patterns
    time_pattern = r'(\d{1,2}):(\d{2})'
    time_matches = re.findall(time_pattern, query)
    if time_matches:
        filters['specific_time'] = time_matches
    
    # Look for duration patterns
    duration_pattern = r'(\d+)\s*(second|minute|hour)s?'
    duration_matches = re.findall(duration_pattern, query_lower)
    if duration_matches:
        filters['duration'] = duration_matches
    
    return filters

def extract_participant_filters(query: str) -> Dict[str, List[str]]:
    """
    Extract participant-based filters from query
    
    Args:
        query: User query string
        
    Returns:
        Dictionary with participant filters
    """
    filters = {}
    
    # Look for extension numbers (4-digit numbers, but not years)
    # Avoid matching years like 2025, 2024, etc.
    extension_pattern = r'\b(?:ext|extension)\s*(\d{4})\b|\b(\d{4})\b(?!\s*(?:year|january|february|march|april|may|june|july|august|september|october|november|december))'
    extension_matches = re.findall(extension_pattern, query, re.IGNORECASE)
    extensions = []
    for match in extension_matches:
        # Extract the matched digits from either capture group
        ext_num = match[0] if match[0] else match[1]
        if ext_num:
            # Additional check: avoid years (1900-2099)
            if not (1900 <= int(ext_num) <= 2099):
                extensions.append(ext_num)
    
    if extensions:
        filters['extensions'] = extensions
    
    # Look for user IDs
    user_pattern = r'\buser\s+(\w+)'
    users = re.findall(user_pattern, query.lower())
    if users:
        filters['users'] = users
    
    # Look for specific participants mentioned
    if 'between' in query.lower():
        # Extract participants in "between X and Y" format
        between_pattern = r'between\s+(\w+)\s+and\s+(\w+)'
        between_match = re.search(between_pattern, query.lower())
        if between_match:
            filters['between'] = [between_match.group(1), between_match.group(2)]
    
    return filters

def build_chroma_filter(query: str) -> Dict:
    """
    Build ChromaDB where filter from natural language query
    
    Args:
        query: User query string
        
    Returns:
        ChromaDB where filter dictionary
    """
    where_filter = {}
    query_lower = query.lower()
    
    # Direction filters
    if 'incoming' in query_lower or 'inbound' in query_lower:
        where_filter['direction'] = 'incoming'
    elif 'outgoing' in query_lower or 'outbound' in query_lower:
        where_filter['direction'] = 'outgoing'
    
    # User filters
    participant_filters = extract_participant_filters(query)
    if 'users' in participant_filters:
        where_filter['user_id'] = participant_filters['users'][0]
    
    # Duration filters
    if 'long' in query_lower or 'longer than' in query_lower:
        where_filter['duration_seconds'] = {'$gte': 30}
    elif 'short' in query_lower or 'brief' in query_lower:
        where_filter['duration_seconds'] = {'$lt': 10}
    
    # Recording filters
    if 'recorded' in query_lower or 'recording' in query_lower:
        where_filter['has_recording'] = True
    elif 'no recording' in query_lower:
        where_filter['has_recording'] = False
    
    # Note: Date filtering is handled differently due to ChromaDB string filtering limitations
    # We'll rely on semantic search and post-processing for date-based queries
    
    return where_filter if where_filter else None
