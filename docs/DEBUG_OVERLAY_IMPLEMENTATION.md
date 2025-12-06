# Debug Overlay Implementation Plan

## Goal
Display real-time debug information on the main screen during initialization (when Aura eye is pulsating), showing module loading status similar to console output.

## Implementation Approach

### 1. Debug Log File Mechanism
- Create a debug log file at `~/LedgerAI/data/aura_init_debug.log`
- main.py writes initialization messages to this file
- GUI reads from this file in real-time

### 2. Debug Overlay Widget
- Create `DebugOverlayWidget` class in `aura_gui.py`
- Transparent overlay with scrolling text
- Positioned at bottom of screen (below Aura eye)
- Shows last N lines of debug messages
- Auto-scrolls as new messages arrive

### 3. Real-time Updates
- GUI polls the debug log file every 500ms during initialization
- Only visible when `_setup_complete == False`
- Automatically hides once initialization is complete

### 4. Message Format
- Display messages as they appear in console
- Strip emoji/formatting for cleaner display
- Show timestamp (optional)
- Color-code by message type (info, warning, error)

## Files to Modify

1. `aura-control/gui/aura_gui.py`
   - Add `DebugOverlayWidget` class
   - Integrate into `AuraGUI` initialization
   - Show/hide based on `_setup_complete` state

2. `aura-control/core/main.py`
   - Add debug log file writer
   - Redirect print statements to both console and log file during initialization
   - Clean up log file after initialization

## Design Considerations

- Overlay should not interfere with Aura eye visibility
- Text should be readable but not overwhelming
- Should automatically hide once initialization completes
- Should handle file read errors gracefully

