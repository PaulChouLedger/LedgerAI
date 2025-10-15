# Automatic Document Ingestion Guide

## Overview

The LedgerAI RAG system now supports **automatic document ingestion** with real-time embedding generation. When you upload a document, it's automatically:
1. Parsed (PDF, DOCX, TXT, MD)
2. Chunked (smart content-aware chunking)
3. Embedded (sentence-transformers)
4. Indexed (FAISS)
5. Ready for queries (instant availability)

## How It Works

### 1. **Immediate Ingestion** (Upload Trigger)
When you upload files through the web interface:
```
Upload File → Auto-Ingest Triggered → Embeddings Built → RAG Updated
```
**Time**: 10-30 seconds (depending on document size)

### 2. **Background Monitoring** (Continuous)
The system monitors `data/input/` every 60 seconds:
```
Check for new files → Process if found → Update RAG → Repeat
```
**Useful for**: Batch uploads, external file drops, automated pipelines

### 3. **Manual Trigger** (On-Demand)
You can also manually rebuild:
```bash
python3 scripts/rebuild_embeddings.py
```

## Supported File Types

| Format | Extension | Status |
|--------|-----------|--------|
| PDF | `.pdf` | ✅ Fully supported |
| Word | `.docx` | ✅ Fully supported |
| Text | `.txt` | ✅ Fully supported |
| Markdown | `.md` | ✅ Fully supported |

## Workflow

### User Uploads a Document:

```
1. User uploads "company_report.pdf" via web interface
   ↓
2. File saved to data/input/company_report.pdf
   ↓
3. Auto-ingest triggered immediately
   ↓
4. PDF parsed → data/parsed/company_report.txt
   ↓
5. Text chunked (smart bio-aware or paragraph-aware)
   ↓
6. Embeddings generated (sentence-transformers)
   ↓
7. FAISS index updated
   ↓
8. Document immediately searchable via RAG
   ↓
9. User can ask: "What's in the company report?"
```

**Total time**: ~10-30 seconds

## Configuration

### Chunking Strategy

Edit `scripts/rebuild_embeddings.py`:

```python
# For team/bio documents
CHUNK_SIZE = 1000
OVERLAP = 200
USE_SMART_CHUNKING = True
DETECT_PERSON_BIOS = True

# For medical/technical docs
USE_SMART_CHUNKING = True
DETECT_PERSON_BIOS = False

# For general content
USE_SMART_CHUNKING = False
DETECT_PERSON_BIOS = False
```

### Monitoring Interval

Edit `aura-control/main.py`:

```python
time.sleep(60)  # Check every 60 seconds
# Change to 30 for faster processing
# Change to 300 (5min) for less frequent checks
```

## State Management

The system tracks processed files in `data/ingest_state.json`:

```json
{
  "processed_files": {
    "document.pdf": {
      "hash": "abc123...",
      "timestamp": "2025-10-07T10:30:00",
      "processed": true
    }
  },
  "last_scan": "2025-10-07T10:30:00"
}
```

- **File modified?** → Re-processed automatically
- **Same file uploaded again?** → Skipped (hash unchanged)
- **File deleted?** → Removed from index on next scan

## Performance

| Document Size | Parse Time | Embed Time | Total Time |
|--------------|------------|------------|------------|
| 10 pages | ~2s | ~3s | ~5s |
| 50 pages | ~5s | ~10s | ~15s |
| 200 pages | ~15s | ~30s | ~45s |

*Times measured on Jetson Orin NX*

## Troubleshooting

### Documents not showing up in search?

1. **Check logs**: Look for `[AutoIngest]` messages
2. **Check state**: `cat data/ingest_state.json`
3. **Manual rebuild**: `python3 scripts/rebuild_embeddings.py`
4. **Restart RAG**: `docker restart aura-rag`

### Auto-ingest not working?

```bash
# Check if auto-ingest is running
ps aux | grep auto_ingest

# Check for errors
tail -f /tmp/aura.log

# Test manually
cd aura-control
python3 auto_ingest.py --help
```

### Files not processing?

- **Check file type**: Must be PDF, DOCX, TXT, or MD
- **Check permissions**: Files must be readable
- **Check file size**: Very large files (>100MB) may timeout
- **Check disk space**: Embeddings need storage

## Advanced Usage

### Process Specific File Types Only

Edit `auto_ingest.py`:

```python
SUPPORTED_EXTENSIONS = {'.pdf', '.docx'}  # Only PDF and Word
```

### Adjust Chunk Size for Specific Domains

```python
# Medical documents (shorter chunks for precision)
chunk_size = 500

# Legal documents (longer chunks for context)
chunk_size = 1500

# Technical docs (medium chunks)
chunk_size = 1000
```

### Disable Auto-Ingest

Comment out in `main.py`:

```python
# Step 7: Start auto-ingest monitoring (if available)
# try:
#     from auto_ingest import AutoIngestPipeline
#     ...
```

## API Integration

Trigger ingestion programmatically:

```python
from auto_ingest import AutoIngestPipeline

# Initialize pipeline
pipeline = AutoIngestPipeline()

# Process new files
pipeline.run_once()

# Or run continuously
pipeline.run_continuous(interval=60)
```

## Best Practices

1. **Upload during low-usage times** for large batches
2. **Use descriptive filenames** for easier tracking
3. **Monitor `ingest_state.json`** to verify processing
4. **Keep original files** as backup
5. **Test with small documents first**
6. **Adjust chunk size** based on document type

## What Happens Behind the Scenes

```
┌─────────────────┐
│  Upload File    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Hash Check      │◄── Is this file new/modified?
└────────┬────────┘
         │ Yes
         ▼
┌─────────────────┐
│ Parse Document  │── PDF → Text, DOCX → Text
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Smart Chunking  │── Bio detection, paragraphs, or fixed
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Generate        │── sentence-transformers on GPU
│ Embeddings      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Update FAISS    │── Add to vector index
│ Index           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Restart RAG     │── Load new index
│ (if needed)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Ready for       │
│ Queries! ✅     │
└─────────────────┘
```

## Summary

✅ **Automatic**: Files processed immediately upon upload  
✅ **Real-time**: Embeddings generated in 10-30 seconds  
✅ **Smart**: Content-aware chunking for better results  
✅ **Universal**: Works with any document type  
✅ **Monitored**: Background scanning for new files  
✅ **Tracked**: State management prevents duplicate processing  

Your RAG system now has **zero-config automatic ingestion**! 🎉

