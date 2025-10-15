# Shared Resources

This directory contains resources shared across multiple containers.

## Files

### `medical_terms.json`
**Single source of truth for medical terminology**

Used by:
- `whisper-container` - Improves transcription accuracy for medical terms
- `llm-container` - Detects medical queries for routing to unified medical mode

**To update medical terms:**
1. Edit `shared/medical_terms.json` (this file only!)
2. Rebuild affected containers:
   ```bash
   docker-compose build aura-whisper aura-llm
   docker-compose up -d
   ```

**DO NOT** create duplicate copies in whisper-container/ or llm-container/
Both Dockerfiles copy from this shared location during build.

---

## Directory Structure

```
shared/
├── README.md              ← This file
└── medical_terms.json     ← Medical terminology (single source of truth)
```

## Why This Matters

Before: Medical terms were duplicated in multiple locations (whisper-container/, llm-container/)
- ❌ Updates required in multiple files
- ❌ Files could get out of sync
- ❌ Maintenance nightmare

Now: Single file in shared/
- ✅ Update once, affects all containers
- ✅ Always synchronized
- ✅ Easy to maintain

---

**Rule:** Any resource used by 2+ containers goes in `shared/`

