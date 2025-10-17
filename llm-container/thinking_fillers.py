"""
Thinking Fillers - Natural phrases to play while LLM generates responses
Used for both TTS (voice) and Telegram (text) to reduce perceived latency

IMPORTANT: Text and audio must be PERFECTLY synchronized
- Each filler has a unique ID
- Text is used for chatbot
- Audio file (if exists) is used for voice
- Both systems output the SAME message
"""
import random
import os
from pathlib import Path

# Base directory for filler audio files
FILLER_AUDIO_DIR = Path(__file__).parent.parent / 'data' / 'fillers' / 'thinking'

# Master list of fillers with IDs and text
# Audio files will be named: {id}.wav
FILLER_LIBRARY = [
    # General thinking fillers
    {'id': 'think_1', 'text': "Let me think..."},
    {'id': 'think_2', 'text': "One moment..."},
    {'id': 'think_3', 'text': "Hmm, let me see..."},
    {'id': 'think_4', 'text': "Give me just a second..."},
    {'id': 'think_5', 'text': "Let me consider that..."},
    {'id': 'think_6', 'text': "Just a moment..."},
    {'id': 'think_7', 'text': "Thinking..."},
    {'id': 'think_8', 'text': "Let me check..."},
    {'id': 'think_9', 'text': "Hold on..."},
    {'id': 'think_10', 'text': "Alright, let me think about that..."},
]

# Context-specific fillers for medical assessment
MEDICAL_FILLERS = {
    'opening': [
        {'id': 'opening_1', 'text': "I understand..."},
        {'id': 'opening_2', 'text': "Okay..."},
        {'id': 'opening_3', 'text': "Alright..."},
        {'id': 'opening_4', 'text': "I see..."},
    ],
    'question_generation': [
        {'id': 'question_1', 'text': "Let me think about what to ask next..."},
        {'id': 'question_2', 'text': "One moment while I consider the best question..."},
        {'id': 'question_3', 'text': "Give me a second to think..."},
        {'id': 'question_4', 'text': "Hmm, let me see..."},
        {'id': 'question_5', 'text': "Let me figure out what to ask..."},
    ],
    'location_clarification': [
        {'id': 'location_1', 'text': "Let me make sure I understand the location..."},
        {'id': 'location_2', 'text': "I want to be more specific about the location..."},
        {'id': 'location_3', 'text': "Let me clarify where exactly..."},
        {'id': 'location_4', 'text': "Hold on, let me get more detail..."},
    ],
    'diagnosis': [
        {'id': 'diagnosis_1', 'text': "Let me review everything..."},
        {'id': 'diagnosis_2', 'text': "Give me a moment to put this together..."},
        {'id': 'diagnosis_3', 'text': "Let me analyze your symptoms..."},
        {'id': 'diagnosis_4', 'text': "Hmm, let me think through this..."},
    ]
}


def get_filler(context='general', use_audio=True):
    """
    Get a random thinking filler
    
    Args:
        context: Type of context ('general', 'opening', 'question_generation', etc.)
        use_audio: If True, will include audio_path if file exists
    
    Returns:
        dict with:
        - 'text': Text version (for chatbot/Telegram/logging)
        - 'audio_path': Path to pre-recorded audio (for TTS/voice) if available
        - 'id': Unique filler ID
    """
    # Get context-specific fillers or default to general
    filler_list = MEDICAL_FILLERS.get(context, FILLER_LIBRARY)
    
    # Select random filler
    filler = random.choice(filler_list)
    
    result = {
        'id': filler['id'],
        'text': filler['text']
    }
    
    # Check if pre-recorded audio exists
    if use_audio:
        audio_path = FILLER_AUDIO_DIR / f"{filler['id']}.wav"
        if audio_path.exists():
            result['audio_path'] = str(audio_path)
    
    return result


def get_filler_text(context='general'):
    """Get just the text filler (for Telegram or logging)"""
    return get_filler(context, use_audio=False)['text']


def get_all_fillers():
    """Get all fillers for audio generation"""
    all_fillers = FILLER_LIBRARY.copy()
    for context_fillers in MEDICAL_FILLERS.values():
        all_fillers.extend(context_fillers)
    return all_fillers


# Quick test
if __name__ == "__main__":
    print("General fillers:")
    for i in range(3):
        filler = get_filler('general')
        print(f"  - [{filler['id']}] {filler['text']}")
        if 'audio_path' in filler:
            print(f"    Audio: {filler['audio_path']}")
    
    print("\nMedical question generation fillers:")
    for i in range(3):
        filler = get_filler('question_generation')
        print(f"  - [{filler['id']}] {filler['text']}")
    
    print("\nAll unique fillers:")
    all_fillers = get_all_fillers()
    print(f"  Total: {len(all_fillers)} fillers")
    for f in all_fillers[:5]:
        print(f"    - [{f['id']}] {f['text']}")

