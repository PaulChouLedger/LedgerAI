# Starting Memory Container

## Quick Start

```bash
cd ~/LedgerAI/setup
docker-compose up -d memory
```

## Check Status

```bash
# Check if container is running
docker ps | grep memory

# Check health
curl http://localhost:11438/health

# View logs
docker logs -f memory-container
```

## Start All Containers (Including Memory)

```bash
cd ~/LedgerAI/setup
docker-compose up -d
```

This will start:
- whisper
- llm-medical (or llm-generic)
- memory ← Memory container

## Stop Memory Container

```bash
cd ~/LedgerAI/setup
docker-compose stop memory
```

## Restart Memory Container

```bash
cd ~/LedgerAI/setup
docker-compose restart memory
```

## Build and Start (First Time)

```bash
cd ~/LedgerAI/setup
docker-compose build memory
docker-compose up -d memory
```

## Verify It's Working

After starting, test with:

```bash
# Health check
curl http://localhost:11438/health

# Store a test conversation
curl -X POST http://localhost:11438/store \
  -H "Content-Type: application/json" \
  -d '{"text": "test conversation", "source": "test"}'

# Check stats
curl http://localhost:11438/stats
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs memory-container

# Check if port is in use
netstat -tuln | grep 11438

# Rebuild container
cd ~/LedgerAI/setup
docker-compose build --no-cache memory
docker-compose up -d memory
```

### Connection Refused

If you see `Connection refused`:
1. Check if container is running: `docker ps | grep memory`
2. If not running, start it: `docker-compose up -d memory`
3. Wait a few seconds for it to initialize
4. Check health: `curl http://localhost:11438/health`

### Port Already in Use

If port 11438 is already in use:
1. Check what's using it: `sudo lsof -i :11438`
2. Stop the conflicting service or change port in docker-compose.yml

