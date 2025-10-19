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
    # General thinking fillers - professional clinician style
    {'id': 'think_1', 'text': "Let me think through your symptoms for a moment..."},
    {'id': 'think_2', 'text': "I need to consider what you've told me so far..."},
    {'id': 'think_3', 'text': "Let me review what we've discussed..."},
    {'id': 'think_4', 'text': "I'm processing the information you've given me..."},
    {'id': 'think_5', 'text': "Let me think about the best way to help you..."},
    {'id': 'think_6', 'text': "I'm considering your symptoms carefully..."},
    {'id': 'think_7', 'text': "Let me take a moment to assess what you've described..."},
    {'id': 'think_8', 'text': "I'm thinking about your situation..."},
    {'id': 'think_9', 'text': "Let me consider the best approach here..."},
    {'id': 'think_10', 'text': "I'm reviewing your symptoms to help you better..."},
    {'id': 'think_11', 'text': "Let me take a moment to understand your condition better..."},
    {'id': 'think_12', 'text': "I'm carefully considering what you've shared with me..."},
    {'id': 'think_13', 'text': "Let me think about how to best address your concerns..."},
    {'id': 'think_14', 'text': "I'm processing your symptoms to provide the best care..."},
    {'id': 'think_15', 'text': "Let me consider the most appropriate next steps..."},
    {'id': 'think_16', 'text': "I'm thinking about your specific situation..."},
    {'id': 'think_17', 'text': "Let me take a moment to evaluate what you've told me..."},
    {'id': 'think_18', 'text': "I'm considering how to best help you with this..."},
    {'id': 'think_19', 'text': "Let me think about the most important aspects of your case..."},
    {'id': 'think_20', 'text': "I'm reviewing your information to provide the best assessment..."},
]

# Context-specific fillers for medical assessment
MEDICAL_FILLERS = {
    'opening': [
        {'id': 'opening_1', 'text': "I understand you're experiencing some discomfort..."},
        {'id': 'opening_2', 'text': "I can see you're concerned about your symptoms..."},
        {'id': 'opening_3', 'text': "I appreciate you sharing that with me..."},
        {'id': 'opening_4', 'text': "I understand this is concerning for you..."},
    ],
    'question_generation': [
        {'id': 'question_1', 'text': "Let me think about what would be most helpful to ask you next..."},
        {'id': 'question_2', 'text': "I need to consider what additional information would help me understand your situation better..."},
        {'id': 'question_3', 'text': "Let me think about the best way to gather more information about your symptoms..."},
        {'id': 'question_4', 'text': "I'm considering what would be most important to know about your condition..."},
        {'id': 'question_5', 'text': "Let me think about what questions would help me better assess your situation..."},
    ],
    'location_clarification': [
        {'id': 'location_1', 'text': "I want to make sure I understand exactly where you're feeling this..."},
        {'id': 'location_2', 'text': "Let me get a clearer picture of the location of your symptoms..."},
        {'id': 'location_3', 'text': "I need to be more specific about where exactly you're experiencing this..."},
        {'id': 'location_4', 'text': "Let me clarify the precise location to better understand your condition..."},
    ],
    'diagnosis': [
        {'id': 'diagnosis_1', 'text': "Let me review everything you've told me so far..."},
        {'id': 'diagnosis_2', 'text': "I'm putting together all the information you've shared with me..."},
        {'id': 'diagnosis_3', 'text': "Let me analyze your symptoms and what you've described..."},
        {'id': 'diagnosis_4', 'text': "I'm considering all the details you've provided to help assess your situation..."},
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
    print(f"  Main library: {len(FILLER_LIBRARY)} fillers")
    for f in all_fillers[:5]:
        print(f"    - [{f['id']}] {f['text']}")

