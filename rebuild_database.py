#!/usr/bin/env python3
"""
Script to rebuild ChromaDB with all CDR records
"""

import os
import shutil
import sys
from pathlib import Path

def rebuild_database():
    """Remove existing ChromaDB and restart the application"""
    
    # Path to ChromaDB directory
    chroma_path = Path("/Users/hiteshvv/Desktop/ragproject/chroma_db")
    
    print("🔄 Rebuilding CDR database...")
    
    # Remove existing ChromaDB directory if it exists
    if chroma_path.exists():
        print(f"🗑️  Removing existing ChromaDB at {chroma_path}")
        shutil.rmtree(chroma_path)
        print("✅ Old database removed")
    
    print("🚀 Database cleared. Ready for fresh rebuild.")
    print("💡 Restart the application to rebuild with all 10,000 records")
    print("\nRun: python app.py")

if __name__ == "__main__":
    rebuild_database()
