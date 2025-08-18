import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

class CDRDataProcessor:
    """Processes CDR (Call Detail Record) data for RAG pipeline"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.records = []
        self.processed_docs = []
    
    def load_data(self) -> List[Dict]:
        """Load CDR data from JSON file"""
        print("Loading CDR data...")
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        
        # Extract hits from Elasticsearch response
        if 'hits' in data and 'hits' in data['hits']:
            self.records = data['hits']['hits']
            print(f"Loaded {len(self.records)} CDR records")
        else:
            raise ValueError("Invalid data format - expected Elasticsearch response")
        
        return self.records
    
    def extract_call_info(self, record: Dict) -> Dict:
        """Extract key call information from a CDR record"""
        source = record.get('_source', {})
        fields = source.get('fields', {})
        
        # Communication record details
        comm_record = fields.get('communicationRecord', {})
        device_info = fields.get('deviceInfo', {})
        participants = fields.get('participants', [])
        com_files = fields.get('comFiles', [])
        
        # Extract participants
        source_participant = None
        destination_participant = None
        for p in participants:
            if p.get('involvement') == 'source':
                source_participant = p.get('address')
            elif p.get('involvement') == 'destination':
                destination_participant = p.get('address')
        
        return {
            'record_id': record.get('_id'),
            'com_uuid': comm_record.get('comUUID'),
            'start_time': comm_record.get('startTime'),
            'end_time': comm_record.get('endTime'),
            'direction': comm_record.get('direction'),
            'com_type': comm_record.get('comType'),
            'active_duration': comm_record.get('activeDuration', 0),
            'source_address': source_participant,
            'destination_address': destination_participant,
            'user_id': device_info.get('pUserID'),
            'channel_id': device_info.get('channelID'),
            'audio_interface': device_info.get('audioInterface'),
            'button_name': device_info.get('buttonName'),
            'platform': comm_record.get('platform', {}).get('name'),
            'file_count': len(com_files),
            'has_audio': any(f.get('fileType') == 'audio' for f in com_files),
            'timestamp': source.get('@timestamp'),
            'timestamp_end': source.get('@timestampend')
        }
    
    def create_searchable_document(self, call_info: Dict) -> str:
        """Create a searchable text document from call information"""
        
        # Format timestamp
        start_time = ""
        if call_info['start_time']:
            try:
                dt = datetime.fromisoformat(call_info['start_time'].replace('Z', '+00:00'))
                start_time = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                start_time = call_info['start_time']
        
        # Calculate duration if available
        duration_text = ""
        if call_info['start_time'] and call_info['end_time']:
            try:
                start_dt = datetime.fromisoformat(call_info['start_time'].replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(call_info['end_time'].replace('Z', '+00:00'))
                duration_seconds = (end_dt - start_dt).total_seconds()
                duration_text = f"Duration: {duration_seconds} seconds"
            except:
                duration_text = f"Active duration: {call_info['active_duration']} seconds"
        
        # Create comprehensive text document
        doc_parts = [
            f"Call Record ID: {call_info['record_id']}",
            f"Communication UUID: {call_info['com_uuid']}",
            f"Call Type: {call_info['com_type']}",
            f"Direction: {call_info['direction']} call",
            f"Start Time: {start_time}",
            duration_text,
            f"Source: {call_info['source_address']} calling Destination: {call_info['destination_address']}",
            f"User: {call_info['user_id']}",
            f"Channel: {call_info['channel_id']}",
            f"Audio Interface: {call_info['audio_interface']}",
            f"Button: {call_info['button_name']}",
            f"Platform: {call_info['platform']}",
            f"Has Audio Recording: {'Yes' if call_info['has_audio'] else 'No'}",
            f"File Count: {call_info['file_count']}"
        ]
        
        # Filter out empty parts
        doc_parts = [part for part in doc_parts if part and not part.endswith(": None") and not part.endswith(": ")]
        
        return " | ".join(doc_parts)
    
    def process_all_records(self) -> List[Dict]:
        """Process all CDR records into searchable documents"""
        print("Processing CDR records...")
        
        self.processed_docs = []
        for i, record in enumerate(self.records):
            try:
                call_info = self.extract_call_info(record)
                searchable_text = self.create_searchable_document(call_info)
                
                processed_doc = {
                    'id': f"cdr_{i}",
                    'text': searchable_text,
                    'metadata': call_info
                }
                
                self.processed_docs.append(processed_doc)
                
            except Exception as e:
                print(f"Error processing record {i}: {e}")
                continue
        
        print(f"Successfully processed {len(self.processed_docs)} documents")
        return self.processed_docs
    
    def get_statistics(self) -> Dict:
        """Get basic statistics about the CDR data"""
        if not self.processed_docs:
            return {}
        
        metadata_list = [doc['metadata'] for doc in self.processed_docs]
        df = pd.DataFrame(metadata_list)
        
        stats = {
            'total_calls': len(df),
            'unique_users': df['user_id'].nunique(),
            'unique_sources': df['source_address'].nunique(),
            'unique_destinations': df['destination_address'].nunique(),
            'call_directions': df['direction'].value_counts().to_dict(),
            'platforms': df['platform'].value_counts().to_dict(),
            'audio_calls': df['has_audio'].sum(),
            'date_range': {
                'start': df['start_time'].min(),
                'end': df['start_time'].max()
            }
        }
        
        return stats

if __name__ == "__main__":
    # Example usage
    processor = CDRDataProcessor("es_data.json")
    records = processor.load_data()
    processed_docs = processor.process_all_records()
    stats = processor.get_statistics()
    
    print("\n=== CDR Data Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print(f"\n=== Sample Processed Document ===")
    if processed_docs:
        print(processed_docs[0]['text'])
