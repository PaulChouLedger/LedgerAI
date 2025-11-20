# Checking XVF3800 Tuning Service

## Quick Status Check

```bash
# Check if service is enabled and running
systemctl status xvf3800-tuning.service
```

## Detailed Diagnostic Commands

### 1. Check Service Status

```bash
# Basic status
systemctl status xvf3800-tuning.service

# Check if service is enabled (runs on boot)
systemctl is-enabled xvf3800-tuning.service

# Check if service is active
systemctl is-active xvf3800-tuning.service
```

### 2. View Service Logs

```bash
# View recent logs (last 50 lines)
journalctl -u xvf3800-tuning.service -n 50

# View logs with timestamps
journalctl -u xvf3800-tuning.service -n 50 --no-pager

# Follow logs in real-time
journalctl -u xvf3800-tuning.service -f

# View logs from last boot
journalctl -u xvf3800-tuning.service -b

# View all logs since service was created
journalctl -u xvf3800-tuning.service --no-pager
```

### 3. Check Service Configuration

```bash
# View service file
cat /etc/systemd/system/xvf3800-tuning.service

# Check when service runs (dependencies)
systemctl show xvf3800-tuning.service | grep -E "After|Before|Wants|Requires"
```

### 4. Test Service Manually

```bash
# Stop the service
sudo systemctl stop xvf3800-tuning.service

# Start the service manually
sudo systemctl start xvf3800-tuning.service

# Restart the service
sudo systemctl restart xvf3800-tuning.service

# Reload service configuration (after editing service file)
sudo systemctl daemon-reload
sudo systemctl restart xvf3800-tuning.service
```

### 5. Check if Tuning Script Runs Successfully

```bash
# Run the tuning script manually (replace USERNAME with your username)
cd ~/LedgerAI
python3 -B setup/scripts/tune_xvf3800.py agc_20_ec

# Check if LEDs are disabled
python3 -B setup/scripts/tune_xvf3800.py show
```

### 6. Verify Device Connection

```bash
# Check if device is detected
lsusb | grep -i "reSpeaker\|XVF3800\|UACDemo"

# Check ALSA devices
arecord -l | grep -i "reSpeaker\|ArrayUAC10\|XVF3800"

# Check if xvf_host binary exists
ls -lh ~/reSpeaker_XVF3800_USB_4MIC_ARRAY/host_control/jetson/xvf_host
```

### 7. Check Service Timing Issues

```bash
# View service start time relative to boot
systemctl show xvf3800-tuning.service -p ActiveEnterTimestamp

# Check if service runs too early (before USB device is ready)
journalctl -u xvf3800-tuning.service -b | grep -i "error\|fail\|not found"
```

## Common Issues and Solutions

### Issue: Service Not Running

**Symptoms:**
- `systemctl status` shows "inactive (dead)"
- LEDs remain on

**Solutions:**
```bash
# Enable and start service
sudo systemctl enable xvf3800-tuning.service
sudo systemctl start xvf3800-tuning.service

# Check logs for errors
journalctl -u xvf3800-tuning.service -n 50
```

### Issue: Service Runs But LEDs Still On

**Symptoms:**
- Service shows "active (exited)" (success)
- But LEDs remain on

**Possible Causes:**
1. **Service runs before device is ready**
   - Solution: Add delay or dependency on USB device

2. **LED commands fail silently**
   - Check logs: `journalctl -u xvf3800-tuning.service | grep -i led`
   - Solution: Run script manually to see errors

3. **Device resets after service runs**
   - Solution: Add a delay or retry mechanism

**Debug:**
```bash
# Run script manually to see LED output
python3 -B ~/LedgerAI/setup/scripts/tune_xvf3800.py agc_20_ec

# Check if LED commands work
~/reSpeaker_XVF3800_USB_4MIC_ARRAY/host_control/jetson/xvf_host LED_BRIGHTNESS 0
```

### Issue: Service Runs Too Early

**Symptoms:**
- Service runs but device not found
- Logs show "device not found" or "xvf_host not found"

**Solution: Add USB device dependency**

Edit service file:
```bash
sudo nano /etc/systemd/system/xvf3800-tuning.service
```

Add to `[Unit]` section:
```
After=sys-subsystem-usb-devices-*.device
Wants=sys-subsystem-usb-devices-*.device
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl restart xvf3800-tuning.service
```

### Issue: Permission Errors

**Symptoms:**
- Logs show "Permission denied" or "cannot access device"

**Solution:**
```bash
# Check service user
systemctl show xvf3800-tuning.service -p User

# Check USB device permissions
ls -l /dev/bus/usb/*/* | grep -i "reSpeaker\|XVF3800"

# Add user to audio group if needed
sudo usermod -aG audio $USER
```

## Quick Diagnostic Script

Save this as `check_xvf_service.sh`:

```bash
#!/bin/bash
echo "🔍 Checking XVF3800 Tuning Service..."
echo ""

echo "1️⃣  Service Status:"
systemctl status xvf3800-tuning.service --no-pager -l | head -10
echo ""

echo "2️⃣  Service Enabled:"
systemctl is-enabled xvf3800-tuning.service
echo ""

echo "3️⃣  Recent Logs:"
journalctl -u xvf3800-tuning.service -n 20 --no-pager | tail -20
echo ""

echo "4️⃣  USB Device:"
lsusb | grep -i "reSpeaker\|XVF3800\|UACDemo" || echo "   ⚠️  Device not found"
echo ""

echo "5️⃣  xvf_host binary:"
if [ -f ~/reSpeaker_XVF3800_USB_4MIC_ARRAY/host_control/jetson/xvf_host ]; then
    echo "   ✅ Found"
    ls -lh ~/reSpeaker_XVF3800_USB_4MIC_ARRAY/host_control/jetson/xvf_host
else
    echo "   ❌ Not found"
fi
echo ""

echo "6️⃣  Test LED disable manually:"
echo "   Run: python3 -B ~/LedgerAI/setup/scripts/tune_xvf3800.py agc_20_ec"
```

Make executable:
```bash
chmod +x check_xvf_service.sh
./check_xvf_service.sh
```

## Fix: Ensure LEDs Are Disabled on Boot

If LEDs are still on, try these steps:

1. **Check service logs for LED commands:**
   ```bash
   journalctl -u xvf3800-tuning.service | grep -i "LED\|led"
   ```

2. **Run script manually to verify:**
   ```bash
   python3 -B ~/LedgerAI/setup/scripts/tune_xvf3800.py agc_20_ec
   ```

3. **If manual run works but service doesn't:**
   - Service may be running before device is ready
   - Add delay to service or change `After` dependency

4. **Add retry logic to service:**
   Edit service file to add a delay:
   ```bash
   sudo nano /etc/systemd/system/xvf3800-tuning.service
   ```
   
   Change ExecStart to:
   ```
   ExecStart=/bin/bash -c 'sleep 5 && __PYTHON_CMD__ -B __LEDGERAI_DIR__/setup/scripts/tune_xvf3800.py agc_20_ec'
   ```

5. **Reload and restart:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart xvf3800-tuning.service
   journalctl -u xvf3800-tuning.service -f
   ```

