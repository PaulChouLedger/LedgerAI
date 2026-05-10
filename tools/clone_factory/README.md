# Aura Clone Factory

Tools for mass-cloning a Jetson Aura puck onto blank NVMe drives. After
each clone boots, an injected one-shot systemd service (`aura-first-boot`)
auto-resets the per-host identity so siblings don't collide.

## Files

- `aura-first-boot.sh` — script run once on first boot of a cloned puck
  (regenerates hostname, machine-id, SSH host keys, Tailscale state)
- `aura-first-boot.service` — systemd unit that invokes the script
- `inject_first_boot.sh` — installs the script + unit onto a block device
  or image file
- `restore_image.sh` — parallel `dd` from one master image to N targets,
  auto-injecting first-boot on each

## Workflow

### Step 1 — capture a master image (once)

After cloning your reference puck's NVMe to a destination disk via:

```
sudo dd if=/dev/sdb of=/dev/sda bs=64M status=progress conv=fsync
```

…return the source NVMe to the puck and keep the cloned destination
plugged in. Capture an image of it to internal storage:

```
sudo dd if=/dev/sda of=/mnt/aura1/master.img bs=64M status=progress
```

(Or skip this step and read directly from the source NVMe each time —
slower, more wear on the puck disk.)

### Step 2 — bake first-boot into the master image (once)

```
sudo ./inject_first_boot.sh /mnt/aura1/master.img
```

Now every restore from this image carries the identity-reset service.

### Step 3 — restore to N blanks (per batch)

Plug as many blank NVMe drives as you have free USB 3 ports. Confirm
which devices they showed up as (`lsblk`). Then:

```
sudo ./restore_image.sh /mnt/aura1/master.img /dev/sdc /dev/sdd /dev/sde
```

USB 3.2 Gen 2 sustains ~700 MB/s per stream, but multiple streams share
the controller — 3 in parallel ≈ 60 min for 1.8 TB each. Use batches
of 3.

### Step 4 — first boot of each new puck

Install the cloned NVMe in the target puck and power on. The
`aura-first-boot` service runs once before `aura4.service`:

- Hostname → `puck-<last 6 hex of MAC>`
- machine-id, SSH host keys, Tailscale state all freshly generated
- `data/app_settings.json` cleared (color scheme picked at first run)
- `data/voice_profiles/` cleared (re-enroll the household)
- The 10k voiceline pool is preserved (hardware-agnostic)

After first boot completes, run on the new puck once:

```
sudo tailscale up
```

…to claim the puck on the tailnet under its new identity. Then set
hostname / color scheme to whatever you want via the GUI.

## Notes

- `aura-first-boot` only runs once. The script disables itself on
  successful completion. If it fails, check `/var/log/aura-first-boot.log`
  on the new puck and re-enable manually.
- All scripts assume the standard Jetson Linux 15-partition layout
  (rootfs is the largest ext4 partition).
