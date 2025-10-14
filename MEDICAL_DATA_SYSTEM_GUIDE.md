# Medical Data Ingestion System Guide

## Overview

This comprehensive medical data ingestion system provides automated collection, processing, and indexing of medical literature for clinician mode RAG (Retrieval-Augmented Generation). The system continuously updates with the latest medical guidelines, research, and clinical information.

## Features

### 🚀 **Automated Data Collection**
- PubMed/MEDLINE article scraping
- Clinical practice guidelines from ACP, AAFP, CDC
- Medical journal articles from NEJM, JAMA, Lancet, BMJ
- Intelligent chunking optimized for medical context
- Medical terminology expansion and synonym handling

### 🏥 **Clinician-Optimized RAG**
- Medical-specific embeddings using BioBERT
- Evidence level assessment
- Clinical relevance scoring
- Medical specialty identification
- Confidence scoring for results

### ⏰ **Automated Updates**
- Daily incremental updates
- Weekly full refreshes
- Email notifications for update status
- Backup and rollback capabilities
- Smart retry and error handling

### 🔧 **Easy Integration**
- Seamless integration with existing RAG system
- Fallback to general RAG when medical data unavailable
- Configurable update schedules
- RESTful API endpoints for integration

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_medical.txt
```

### 2. Initial Setup

```bash
# Run initial medical data ingestion
python3 medical_data_ingestion.py --update

# Verify installation
python3 medical_data_ingestion.py --stats
```

### 3. Start Automated Updates

```bash
# For daily updates (recommended for production)
python3 medical_update_scheduler.py --schedule-daily

# For weekly updates only
python3 medical_update_scheduler.py --schedule-weekly

# Manual one-time update
python3 medical_update_scheduler.py --run-once
```

### 4. Integrate with Clinician Mode

```python
from clinician_rag import search_clinician_info

# Use in clinician mode
medical_response = search_clinician_info("What are the latest diabetes management guidelines?")
print(medical_response)
```

## Configuration

### Environment Variables

Set these environment variables for enhanced functionality:

```bash
# PubMed API (optional, improves rate limits)
export PUBMED_API_KEY="your_pubmed_api_key"

# Email notifications (optional)
export MEDICAL_UPDATE_EMAIL_USER="your_email@gmail.com"
export MEDICAL_UPDATE_EMAIL_PASSWORD="your_app_password"

# Medical model (optional, falls back gracefully)
export MEDICAL_MODEL="pritamdeka/BioBERT-mnli-snli-scinli-stsb"
```

### Configuration Files

#### Medical Scheduler Config (`medical_scheduler_config.json`)

```json
{
  "email_notifications": true,
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587,
  "notification_recipients": ["admin@yourorg.com"],
  "daily_update_time": "02:00",
  "weekly_update_day": "sunday",
  "weekly_update_time": "03:00",
  "max_retries": 3,
  "retry_delay": 300,
  "enable_incremental_updates": true,
  "backup_before_update": true
}
```

## Data Sources

### 📚 **PubMed/MEDLINE**
- Latest research articles and clinical trials
- Abstracts and metadata extraction
- Author and journal information
- DOI and PMID references

### 📋 **Clinical Guidelines**
- **ACP** (American College of Physicians)
- **AAFP** (American Academy of Family Physicians)
- **CDC** (Centers for Disease Control and Prevention)
- Evidence-based recommendations

### 🏥 **Medical Journals**
- **NEJM** (New England Journal of Medicine)
- **JAMA** (Journal of the American Medical Association)
- **The Lancet**
- **BMJ** (British Medical Journal)

## Medical Data Structure

```
data/medical/
├── sources/           # PubMed articles and research
│   ├── pmid_123456.json
│   └── pmid_789012.json
├── guidelines/        # Clinical practice guidelines
│   ├── acp_guideline_001.json
│   └── aafp_recommendation_002.json
├── journals/          # Medical journal articles
│   ├── nejm_article_001.json
│   └── jama_article_002.json
├── embeddings/        # FAISS index and vectors
│   ├── medical_index.faiss
│   ├── medical_vectors.npy
│   ├── medical_chunks.npy
│   └── medical_metadata.json
├── backups/           # Automatic backups
│   └── medical_backup_20241201_020000/
└── ingestion_state.json   # Update tracking
```

## API Reference

### Medical Data Ingester

```python
from medical_data_ingestion import MedicalDataIngester

ingester = MedicalDataIngester()

# Run full ingestion pipeline
ingester.run_full_ingestion(force_update=True)

# Get statistics
stats = ingester.get_medical_stats()
print(f"Total documents: {stats['total_documents']}")
```

### Clinician RAG

```python
from clinician_rag import ClinicianRAG

clinician_rag = ClinicianRAG()

# Search medical information
results = clinician_rag.search_medical_info("diabetes treatment", k=5)

# Get formatted context
context = clinician_rag.get_medical_context("diabetes treatment", results)
```

### Update Scheduler

```python
from medical_update_scheduler import MedicalUpdateScheduler

scheduler = MedicalUpdateScheduler()

# Run manual update
result = scheduler.run_medical_update('daily')

# Get scheduler status
status = scheduler.get_scheduler_status()
```

## Advanced Usage

### Custom Medical Queries

The system supports various medical query types:

```python
# Clinical guidelines
"What are the current guidelines for hypertension management?"

# Treatment protocols
"How do you treat acute myocardial infarction?"

# Diagnostic criteria
"What are the symptoms of pneumonia?"

# Latest research
"Latest advances in cancer immunotherapy"

# Drug information
"Contraindications for metformin"

# Emergency protocols
"Management of anaphylactic shock"
```

### Medical Terminology Expansion

The system automatically expands queries with medical synonyms:

- "Heart attack" → "Myocardial infarction"
- "High blood pressure" → "Hypertension"
- "Diabetes" → "Diabetes mellitus"
- "Stroke" → "Cerebrovascular accident"

### Evidence-Based Responses

All responses include:
- **Evidence Level**: High/Moderate/Low/Guideline
- **Clinical Relevance**: High/Moderate/Low
- **Medical Specialties**: Cardiology, Endocrinology, etc.
- **Confidence Score**: 0.0-1.0

## Troubleshooting

### Common Issues

#### 1. **No Medical Data Found**
```bash
# Check if medical data exists
python3 medical_data_ingestion.py --stats

# Run initial ingestion if empty
python3 medical_data_ingestion.py --update --force
```

#### 2. **Embedding Generation Fails**
```bash
# Check GPU availability
python3 -c "import torch; print(torch.cuda.is_available())"

# Use CPU fallback if needed
export CUDA_VISIBLE_DEVICES=""
python3 medical_data_ingestion.py --rebuild
```

#### 3. **Scheduler Not Running**
```bash
# Check scheduler status
python3 medical_update_scheduler.py --status

# Manual trigger
python3 medical_update_scheduler.py --run-once
```

#### 4. **Email Notifications Not Working**
```bash
# Check environment variables
echo $MEDICAL_UPDATE_EMAIL_USER
echo $MEDICAL_UPDATE_EMAIL_PASSWORD

# Test email configuration
python3 -c "
import smtplib
# Test SMTP connection manually
"
```

### Performance Optimization

#### Memory Usage
```bash
# Monitor memory usage during updates
python3 -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

#### Update Frequency Tuning
```json
{
  "daily_update_time": "01:00",      // Off-peak hours
  "weekly_update_day": "sunday",     // Weekend for full updates
  "enable_incremental_updates": true // Faster daily updates
}
```

## Security Considerations

### API Keys
- Store PubMed API keys securely in environment variables
- Rotate API keys regularly
- Monitor API usage limits

### Data Privacy
- Medical data is stored locally only
- No patient data or PHI (Protected Health Information) processed
- All data is for educational/clinical reference only

### Network Security
- Use HTTPS for all external API calls
- Implement rate limiting for web scraping
- Monitor for unusual network activity

## Monitoring and Maintenance

### Health Checks

```bash
# Daily health check script
#!/bin/bash
python3 medical_data_ingestion.py --stats
python3 medical_update_scheduler.py --status

# Check for failed updates
python3 -c "
import json
with open('data/medical/update_state.json') as f:
    state = json.load(f)
    print(f'Failed updates: {state.get(\"failed_updates\", 0)}')
"
```

### Log Rotation

```bash
# Rotate medical data logs weekly
find data/medical -name "*.log" -mtime +7 -delete

# Archive old backups
find data/medical/backups -name "medical_backup_*" -mtime +30 -exec rm -rf {} \;
```

### Backup Strategy

```bash
# Automated backup verification
python3 -c "
import os
from pathlib import Path

backup_dir = Path('data/medical/backups')
if backup_dir.exists():
    backups = list(backup_dir.glob('medical_backup_*'))
    print(f'Available backups: {len(backups)}')
    if backups:
        latest = max(backups, key=lambda x: x.stat().st_mtime)
        print(f'Latest backup: {latest.name}')
"
```

## Integration Examples

### Web Application Integration

```python
# Flask endpoint for clinician queries
from flask import Flask, request, jsonify
from clinician_rag import search_clinician_info

app = Flask(__name__)

@app.route('/api/clinician/search', methods=['POST'])
def clinician_search():
    data = request.get_json()
    query = data.get('query', '')
    k = data.get('k', 5)

    result = search_clinician_info(query, k)
    return jsonify({'response': result})

if __name__ == '__main__':
    app.run(port=5000)
```

### Command Line Integration

```bash
# Interactive medical query tool
#!/bin/bash
echo "Medical Query Tool"
read -p "Enter your medical question: " query

python3 -c "
from clinician_rag import search_clinician_info
result = search_clinician_info('$query')
print(result)
"
```

## Future Enhancements

### Planned Features
1. **FDA Drug Database Integration**
2. **WHO Global Guidelines**
3. **Medical Image Analysis**
4. **Multi-language Support**
5. **Real-time Clinical Trial Updates**
6. **Integration with EHR Systems**

### Extension Points
- Custom medical data sources
- Specialized medical models
- Domain-specific ontologies
- Advanced evidence grading

## Support and Contributing

### Getting Help
1. Check the troubleshooting section
2. Review log files in `data/medical/`
3. Monitor system resource usage
4. Verify network connectivity for external sources

### Contributing
1. Follow existing code patterns
2. Add comprehensive tests
3. Update documentation
4. Consider backward compatibility

## License and Compliance

This medical data ingestion system is designed for:
- Educational purposes
- Clinical reference (with appropriate disclaimers)
- Research and development

**Important Disclaimers:**
- Not for direct patient care without professional oversight
- Always consult current clinical guidelines
- Not a substitute for professional medical judgment
- Evidence levels provided for informational purposes only

---

For questions or issues, please refer to the troubleshooting section or contact the development team.
