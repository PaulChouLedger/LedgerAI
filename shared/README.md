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
2. Restart affected containers (NO rebuild needed!):
   ```bash
   docker-compose restart whisper llm
   ```

**DO NOT** create duplicate copies anywhere else.
The `shared/` directory is mounted into containers at runtime via docker-compose.yml:
```yaml
volumes:
  - ../shared:/shared
```

Changes are **immediately available** when containers restart!

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
- ❌ Required container rebuild to update
- ❌ Maintenance nightmare

Now: Single file in shared/ (mounted into containers)
- ✅ Update once, affects all containers
- ✅ Always synchronized
- ✅ Changes take effect on restart (no rebuild!)
- ✅ Easy to maintain

---

**Rule:** Any resource used by 2+ containers goes in `shared/`

