# Quick Test Reference

## Fastest Way to Test

```bash
cd chatterbox-container
python3 test_api_only.py
```

## If Container Not Running

```bash
# Start container
cd ../setup
docker compose up -d chatterbox-tts

# Wait a few seconds, then test
cd ../chatterbox-container
python3 test_api_only.py
```

## Full Test (Build + Start + Test)

```bash
cd chatterbox-container
python3 test_independent.py
```

## Manual Quick Check

```bash
# Health
curl http://localhost:11437/health

# Synthesis
curl -X POST http://localhost:11437/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "test"}' \
  --output test.wav
```

## Check Container Status

```bash
docker ps | grep chatterbox
docker logs chatterbox-tts
```

## Remote Container Testing

```bash
export CHATTERBOX_URL=http://remote-host:11437
python3 test_api_only.py
```
