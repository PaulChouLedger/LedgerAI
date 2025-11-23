# Remove Wyoming Container

## Stop and Remove Wyoming Container

```bash
# Stop the container
docker stop setup-wyoming-openwakeword-1

# Remove the container
docker rm setup-wyoming-openwakeword-1

# Or stop and remove in one command
docker stop setup-wyoming-openwakeword-1 && docker rm setup-wyoming-openwakeword-1
```

## Verify Removal

```bash
# Check if container is gone
docker ps -a | grep wyoming

# Should return nothing if successfully removed
```

## Start Memory Container

After removing Wyoming, start the memory container:

```bash
cd ~/LedgerAI/setup
docker-compose up -d memory
```

## Verify Memory Container is Running

```bash
# Check status
docker ps | grep memory

# Check health
curl http://localhost:11438/health

# View logs
docker logs -f memory-container
```

## Note

The references to "OpenWakeWord" in the codebase are for the **Python library** (not Wyoming container), which is used directly in the code. These should **NOT** be removed as they are the active wake word detection system.

Wyoming was a separate container-based approach that is no longer used.

