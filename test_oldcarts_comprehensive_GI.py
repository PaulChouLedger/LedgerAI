#!/usr/bin/env python3
"""
Comprehensive OLDCARTS Normalization Test for GI System
Tests 500 patient prompts with majority focusing on UK language variations
"""

import sys
import os
import json
import re
import time
from typing import List, Dict, Tuple

def test_oldcarts_comprehensive_gi():
    """Comprehensive test of OLDCARTS normalization for GI system with 500 patient prompts (UK language focus)"""
    print("🧪 Comprehensive OLDCARTS Normalization Test - GI System")
    print("=" * 70)
    
    # Find the synonyms file
    synonym_file = None
    possible_paths = [
        'llm-container/synonyms/gi_synonyms_oldcarts.json',
        './llm-container/synonyms/gi_synonyms_oldcarts.json',
        'synonyms/gi_synonyms_oldcarts.json'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            synonym_file = path
            print(f"✅ Found synonyms file: {path}")
            break
    
    if not synonym_file:
        print("❌ Could not find gi_synonyms_oldcarts.json file")
        return False
    
    # Load the synonyms
    try:
        with open(synonym_file, 'r') as f:
            oldcarts_synonyms = json.load(f)
        print(f"✅ Successfully loaded OLDCARTS synonyms")
        print(f"   Found {len(oldcarts_synonyms)} main categories")
    except Exception as e:
        print(f"❌ Failed to load synonyms: {e}")
        return False
    
    # Flatten OLDCARTS structure into standard_term -> variations mapping
    synonyms = {}
    for category, subcategories in oldcarts_synonyms.items():
        if isinstance(subcategories, dict):
            for subcategory, variations in subcategories.items():
                if isinstance(variations, list):
                    # Create standard term from category and subcategory
                    standard_term = f"{category}_{subcategory}".replace("_", " ")
                    synonyms[standard_term] = variations
                elif isinstance(variations, dict):
                    # Handle nested structures
                    for nested_key, nested_variations in variations.items():
                        if isinstance(nested_variations, list):
                            standard_term = f"{category}_{subcategory}_{nested_key}".replace("_", " ")
                            synonyms[standard_term] = nested_variations
        elif isinstance(subcategories, list):
            # Direct list of variations
            standard_term = category.replace("_", " ")
            synonyms[standard_term] = subcategories
    
    print(f"   Loaded {len(synonyms)} synonym categories with {sum(len(v) for v in synonyms.values())} total variations")
    
    # Normalization function
    def normalize_text(text):
        """Normalize text using OLDCARTS synonyms"""
        normalized_text = text.lower()
        
        # Apply synonym replacements
        all_variations = []
        for standard_term, variations in synonyms.items():
            for variation in variations:
                all_variations.append((len(variation), variation, standard_term))
        
        # Sort by length (longest first) to avoid partial replacements
        all_variations.sort(key=lambda x: x[0], reverse=True)
        
        matches_found = []
        for length, variation, standard_term in all_variations:
            pattern = r'\b' + re.escape(variation) + r'\b'
            if re.search(pattern, normalized_text, re.IGNORECASE):
                normalized_text = re.sub(pattern, standard_term, normalized_text, flags=re.IGNORECASE)
                matches_found.append((variation, standard_term))
        
        return normalized_text, matches_found
    
    # Generate comprehensive test cases
    def generate_comprehensive_test_cases():
        """Generate 500 comprehensive test cases with UK language focus"""
        
        # Base templates for different types of complaints
        base_templates = [
            # Location-based complaints
            "my {location} hurts",
            "I have pain in my {location}",
            "there's a pain in my {location}",
            "my {location} is aching",
            "I've got a pain in my {location}",
            "my {location} is sore",
            "I feel pain in my {location}",
            "there's discomfort in my {location}",
            "my {location} is tender",
            "I've got an ache in my {location}",
            
            # UK-specific expressions
            "my {location} is playing up",
            "I've got a dodgy {location}",
            "my {location} is giving me gyp",
            "there's something wrong with my {location}",
            "my {location} is bothering me",
            "I've got a bit of bother with my {location}",
            "my {location} is niggling",
            "I've got a twinge in my {location}",
            
            # Character + Location combinations
            "I have {character} pain in my {location}",
            "there's a {character} ache in my {location}",
            "my {location} has a {character} pain",
            "I've got {character} pain in my {location}",
            "my {location} is {character}",
            
            # Onset + Location combinations
            "my {location} started hurting {onset}",
            "I've had pain in my {location} {onset}",
            "my {location} began to ache {onset}",
            "the pain in my {location} came on {onset}",
            
            # Duration + Location combinations
            "I've had pain in my {location} for {duration}",
            "my {location} has been hurting for {duration}",
            "I've been having {location} pain for {duration}",
            "my {location} has been aching for {duration}",
            
            # Aggravating factors
            "my {location} hurts when I {aggravating}",
            "the pain in my {location} gets worse when I {aggravating}",
            "my {location} is worse after {aggravating}",
            "I get pain in my {location} when I {aggravating}",
            
            # Alleviating factors
            "my {location} feels better when I {alleviating}",
            "the pain in my {location} improves with {alleviating}",
            "my {location} is better when I {alleviating}",
            "I feel better when I {alleviating}",
            
            # Radiation patterns
            "the pain in my {location} goes to my {radiation}",
            "my {location} hurts and it radiates to my {radiation}",
            "I have pain in my {location} that spreads to my {radiation}",
            "the ache in my {location} travels to my {radiation}",
            
            # Timing patterns
            "my {location} pain is {timing}",
            "the pain in my {location} comes and goes {timing}",
            "my {location} hurts {timing}",
            "I get {timing} pain in my {location}",
            
            # Severity descriptions
            "I have {severity} pain in my {location}",
            "my {location} is {severity} painful",
            "the pain in my {location} is {severity}",
            "I've got {severity} pain in my {location}",
            
            # Associated symptoms
            "I have {location} pain and I feel {symptom}",
            "my {location} hurts and I'm {symptom}",
            "I've got pain in my {location} and I'm {symptom}",
            "my {location} is aching and I feel {symptom}",
            
            # Complex combinations
            "I have {character} pain in my {location} that started {onset} and gets worse when I {aggravating}",
            "my {location} has been {character} for {duration} and it radiates to my {radiation}",
            "I've got {severity} pain in my {location} that comes and goes {timing}",
            "my {location} hurts {character} and I feel {symptom}",
        ]
        
        # Fill-in values for templates
        fill_values = {
            'location': [
                # Common patient terms
                'tummy', 'belly', 'stomach', 'gut', 'abdomen', 'tummy area', 'belly area',
                'upper stomach', 'lower stomach', 'middle stomach', 'top of stomach',
                'upper belly', 'lower belly', 'middle belly', 'top of belly',
                'upper tummy', 'lower tummy', 'middle tummy', 'top of tummy',
                'left side', 'right side', 'left part', 'right part',
                'left upper', 'right upper', 'left lower', 'right lower',
                'upper left', 'upper right', 'lower left', 'lower right',
                'middle left', 'middle right', 'center', 'middle',
                'under ribs', 'below ribs', 'above belly button', 'below belly button',
                'near hip', 'near pelvis', 'groin area', 'flank area',
                'behind belly button', 'around belly button', 'near belly button',
                'chest area', 'chest', 'breastbone area', 'behind breastbone',
                'shoulder area', 'back area', 'side area',
                
                # UK-specific terms
                'tummy area', 'belly area', 'gut area', 'stomach area',
                'left side of tummy', 'right side of tummy', 'left side of belly',
                'right side of belly', 'left side of stomach', 'right side of stomach',
                'top part of tummy', 'bottom part of tummy', 'top part of belly',
                'bottom part of belly', 'top part of stomach', 'bottom part of stomach',
            ],
            
            'character': [
                'sharp', 'dull', 'burning', 'cramping', 'throbbing', 'stabbing',
                'aching', 'sore', 'tender', 'heavy', 'pressure', 'squeezing',
                'tight', 'full', 'bloated', 'swollen', 'distended',
                'piercing', 'cutting', 'knife-like', 'fire-like', 'hot',
                'cold', 'numb', 'tingling', 'pins and needles',
                'pulsating', 'beating', 'rhythmic', 'wavelike', 'colicky',
                'gnawing', 'grinding', 'crushing', 'splitting', 'shooting',
                'electric', 'stinging', 'biting', 'prickling',
                
                # UK-specific terms
                'niggling', 'dodgy', 'playing up', 'giving me gyp',
                'bothering me', 'annoying', 'irritating', 'uncomfortable',
                'unpleasant', 'nasty', 'horrible', 'awful', 'terrible',
            ],
            
            'onset': [
                'suddenly', 'all of a sudden', 'out of nowhere', 'without warning',
                'quickly', 'rapidly', 'fast', 'immediately', 'instantly',
                'gradually', 'slowly', 'over time', 'bit by bit', 'step by step',
                'progressively', 'getting worse', 'worsening', 'increasing',
                'yesterday', 'this morning', 'last night', 'earlier today',
                'a few hours ago', 'a few days ago', 'last week', 'recently',
                'after eating', 'after drinking', 'after exercise', 'after work',
                'when I woke up', 'when I got up', 'when I went to bed',
            ],
            
            'duration': [
                'a few minutes', 'a few hours', 'a few days', 'a few weeks',
                'several minutes', 'several hours', 'several days', 'several weeks',
                'about an hour', 'about a day', 'about a week', 'about a month',
                'since yesterday', 'since this morning', 'since last night',
                'since last week', 'since last month', 'for a while now',
                'on and off', 'intermittently', 'sporadically', 'occasionally',
                'constantly', 'all the time', 'continuously', 'non-stop',
                'most of the time', 'most days', 'every day', 'daily',
                'every few hours', 'every few days', 'weekly', 'monthly',
            ],
            
            'aggravating': [
                'eat', 'drink', 'move', 'walk', 'bend', 'twist', 'turn',
                'breathe deeply', 'cough', 'sneeze', 'laugh', 'cry',
                'lie down', 'sit up', 'stand up', 'get up', 'move around',
                'exercise', 'work out', 'lift things', 'carry things',
                'stress', 'worry', 'get anxious', 'get nervous',
                'eat spicy food', 'eat fatty food', 'eat too much',
                'drink alcohol', 'drink coffee', 'drink cold drinks',
                'touch it', 'press on it', 'massage it', 'rub it',
                'wear tight clothes', 'wear a belt', 'sit for long time',
                'stand for long time', 'drive', 'travel', 'fly',
            ],
            
            'alleviating': [
                'rest', 'lie down', 'sit down', 'stand up', 'walk around',
                'move around', 'stretch', 'massage it', 'rub it', 'press on it',
                'apply heat', 'apply cold', 'use a heating pad', 'use ice',
                'take medicine', 'take painkillers', 'take medication',
                'eat something', 'drink something', 'drink water',
                'breathe deeply', 'relax', 'meditate', 'distract myself',
                'change position', 'adjust my posture', 'wear loose clothes',
                'avoid certain foods', 'avoid certain activities',
                'sleep', 'nap', 'take a break', 'go for a walk',
            ],
            
            'radiation': [
                'back', 'chest', 'shoulder', 'arm', 'leg', 'neck', 'head',
                'right shoulder', 'left shoulder', 'right arm', 'left arm',
                'right leg', 'left leg', 'right side', 'left side',
                'upper back', 'lower back', 'middle back', 'spine',
                'chest area', 'breastbone', 'ribs', 'flank', 'groin',
                'thigh', 'knee', 'ankle', 'foot', 'hand', 'wrist',
                'jaw', 'ear', 'eye', 'forehead', 'temple',
            ],
            
            'timing': [
                'constantly', 'all the time', 'continuously', 'non-stop',
                'intermittently', 'on and off', 'comes and goes', 'sporadically',
                'occasionally', 'sometimes', 'periodically', 'regularly',
                'in the morning', 'at night', 'during the day', 'in the evening',
                'when I wake up', 'when I go to bed', 'after meals',
                'before meals', 'when I\'m hungry', 'when I\'m full',
                'when I\'m stressed', 'when I\'m relaxed', 'when I\'m active',
                'when I\'m resting', 'when I\'m lying down', 'when I\'m sitting',
                'when I\'m standing', 'when I\'m walking', 'when I\'m driving',
            ],
            
            'severity': [
                'mild', 'slight', 'minor', 'not too bad', 'manageable',
                'moderate', 'medium', 'somewhat', 'noticeable', 'uncomfortable',
                'severe', 'bad', 'really bad', 'terrible', 'awful', 'horrible',
                'excruciating', 'unbearable', 'intense', 'extreme', 'agonizing',
                'mild to moderate', 'moderate to severe', 'mild to severe',
                '1 out of 10', '2 out of 10', '3 out of 10', '4 out of 10',
                '5 out of 10', '6 out of 10', '7 out of 10', '8 out of 10',
                '9 out of 10', '10 out of 10', 'worst pain ever',
                'barely noticeable', 'very mild', 'quite mild',
                'quite severe', 'very severe', 'extremely severe',
                
                # UK-specific severity terms
                'not too bad', 'not great', 'pretty bad', 'quite bad',
                'really quite bad', 'absolutely terrible', 'bloody awful',
                'not brilliant', 'not good', 'pretty rubbish', 'quite nasty',
            ],
            
            'symptom': [
                'nauseous', 'queasy', 'sick', 'dizzy', 'lightheaded',
                'weak', 'tired', 'exhausted', 'fatigued', 'lethargic',
                'feverish', 'hot', 'cold', 'sweaty', 'clammy',
                'bloated', 'full', 'gassy', 'constipated', 'diarrhea',
                'vomiting', 'throwing up', 'puking', 'retching',
                'heartburn', 'acid reflux', 'indigestion', 'upset stomach',
                'loss of appetite', 'no appetite', 'can\'t eat', 'don\'t want to eat',
                'thirsty', 'dehydrated', 'dry mouth', 'bitter taste',
                'anxious', 'worried', 'stressed', 'panicky', 'restless',
                'irritable', 'moody', 'cranky', 'grumpy', 'depressed',
                'confused', 'foggy', 'unclear', 'disoriented',
                
                # UK-specific symptom terms
                'not quite right', 'off colour', 'under the weather',
                'not myself', 'not feeling great', 'feeling rough',
                'feeling poorly', 'feeling unwell', 'feeling rubbish',
                'feeling awful', 'feeling terrible', 'feeling horrible',
                'not up to much', 'not up to par', 'not 100%',
                'a bit peaky', 'a bit poorly', 'a bit rough',
            ],
        }
        
        # Generate test cases
        test_cases = []
        
        # Add reduced simple location-based cases (focus on UK terms)
        for location in fill_values['location'][:8]:  # Reduced from 20 to focus on UK expressions
            test_cases.extend([
                f"my {location} hurts",
                f"I have pain in my {location}",
                f"there's a pain in my {location}",
                f"my {location} is aching",
                f"I've got a pain in my {location}",
            ])
        
        # Add reduced character + location combinations (focus on UK terms)
        for character in fill_values['character'][:5]:  # Reduced from 10
            for location in fill_values['location'][:5]:  # Reduced from 10
                test_cases.extend([
                    f"I have {character} pain in my {location}",
                    f"there's a {character} ache in my {location}",
                ])
        
        # Add reduced onset combinations
        for onset in fill_values['onset'][:5]:  # Reduced from 10
            for location in fill_values['location'][:5]:  # Reduced from 10
                test_cases.extend([
                    f"my {location} started hurting {onset}",
                    f"I've had pain in my {location} {onset}",
                ])
        
        # Add reduced duration combinations
        for duration in fill_values['duration'][:5]:  # Reduced from 10
            for location in fill_values['location'][:5]:  # Reduced from 10
                test_cases.extend([
                    f"I've had pain in my {location} for {duration}",
                    f"my {location} has been hurting for {duration}",
                ])
        
        # Add reduced aggravating factor combinations
        for aggravating in fill_values['aggravating'][:5]:  # Reduced from 10
            for location in fill_values['location'][:5]:  # Reduced from 10
                test_cases.extend([
                    f"my {location} hurts when I {aggravating}",
                    f"the pain in my {location} gets worse when I {aggravating}",
                ])
        
        # Add reduced alleviating factor combinations
        for alleviating in fill_values['alleviating'][:5]:  # Reduced from 10
            for location in fill_values['location'][:5]:  # Reduced from 10
                test_cases.extend([
                    f"my {location} feels better when I {alleviating}",
                    f"the pain in my {location} improves with {alleviating}",
                ])
        
        # Add reduced radiation combinations
        for radiation in fill_values['radiation'][:5]:  # Reduced from 10
            for location in fill_values['location'][:5]:  # Reduced from 10
                test_cases.extend([
                    f"the pain in my {location} goes to my {radiation}",
                    f"my {location} hurts and it radiates to my {radiation}",
                ])
        
        # Add reduced timing combinations
        for timing in fill_values['timing'][:5]:  # Reduced from 10
            for location in fill_values['location'][:5]:  # Reduced from 10
                test_cases.extend([
                    f"my {location} pain is {timing}",
                    f"the pain in my {location} comes and goes {timing}",
                ])
        
        # Add reduced severity combinations
        for severity in fill_values['severity'][:5]:  # Reduced from 10
            for location in fill_values['location'][:5]:  # Reduced from 10
                test_cases.extend([
                    f"I have {severity} pain in my {location}",
                    f"my {location} is {severity} painful",
                ])
        
        # Add reduced associated symptom combinations
        for symptom in fill_values['symptom'][:5]:  # Reduced from 10
            for location in fill_values['location'][:5]:  # Reduced from 10
                test_cases.extend([
                    f"I have {location} pain and I feel {symptom}",
                    f"my {location} hurts and I'm {symptom}",
                ])
        
        # Add reduced complex multi-factor combinations
        complex_combinations = [
            "I have sharp pain in my upper right that started suddenly after eating",
            "my left lower belly has been cramping for a few days and it radiates to my back",
            "I've got severe pain in my tummy that comes and goes constantly",
            "my stomach hurts when I eat and I feel nauseous",
            "there's a burning pain in my chest that gets worse when I lie down",
        ]
        test_cases.extend(complex_combinations)
        
        # Add extensive UK-specific expressions (majority of test cases)
        uk_expressions = [
            # Basic UK expressions
            "my tummy is playing up",
            "I've got a dodgy belly",
            "my stomach is giving me gyp",
            "there's something wrong with my gut",
            "my tummy is bothering me",
            "I've got a bit of bother with my belly",
            "my stomach is niggling",
            "I've got a twinge in my tummy",
            "my belly is not quite right",
            "I'm feeling a bit peaky in my stomach",
            "my gut is off colour",
            "I'm not feeling great in my tummy area",
            "my belly is feeling rough",
            "I've got a nasty pain in my stomach",
            "my tummy is absolutely terrible",
            "I feel rubbish in my belly",
            "my gut is bloody awful",
            "my stomach is not brilliant",
            "I've got a horrible pain in my tummy",
            "my belly is pretty rubbish",
            
            # More UK expressions with locations
            "my upper tummy is playing up",
            "I've got a dodgy lower belly",
            "my left side tummy is giving me gyp",
            "my right side belly is bothering me",
            "my middle tummy is niggling",
            "I've got a twinge in my upper belly",
            "my lower tummy is not quite right",
            "I'm feeling a bit peaky in my left belly",
            "my right tummy is off colour",
            "I'm not feeling great in my lower belly area",
            "my upper belly is feeling rough",
            "I've got a nasty pain in my right tummy",
            "my left belly is absolutely terrible",
            "I feel rubbish in my upper tummy",
            "my lower gut is bloody awful",
            "my middle stomach is not brilliant",
            "I've got a horrible pain in my left tummy",
            "my right belly is pretty rubbish",
            
            # UK expressions with timing
            "my tummy has been playing up for days",
            "I've had a dodgy belly since this morning",
            "my stomach has been giving me gyp all week",
            "my gut has been bothering me on and off",
            "my belly has been niggling for hours",
            "I've had a twinge in my tummy since yesterday",
            "my stomach has been not quite right for ages",
            "I've been feeling a bit peaky in my belly lately",
            "my gut has been off colour since last night",
            "my tummy has been feeling rough all day",
            "I've had a nasty pain in my belly for weeks",
            "my stomach has been absolutely terrible since Monday",
            "I've been feeling rubbish in my tummy for days",
            "my gut has been bloody awful since the weekend",
            "my belly has been not brilliant for a while",
            "I've had a horrible pain in my tummy since this afternoon",
            "my stomach has been pretty rubbish for months",
            
            # UK expressions with severity
            "my tummy is a bit dodgy",
            "I've got a really dodgy belly",
            "my stomach is quite niggling",
            "my gut is really bothering me",
            "my belly is quite painful",
            "I've got a really nasty pain in my tummy",
            "my stomach is quite terrible",
            "my gut is really rubbish",
            "my belly is quite awful",
            "my tummy is really not brilliant",
            "I've got a quite horrible pain in my belly",
            "my stomach is really pretty rubbish",
            
            # UK expressions with character
            "my tummy is a bit achy",
            "I've got a sharp pain in my dodgy belly",
            "my stomach is quite crampy",
            "my gut is really burning",
            "my belly is quite stabbing",
            "I've got a really gnawing pain in my tummy",
            "my stomach is quite throbbing",
            "my gut is really shooting",
            "my belly is quite stinging",
            "my tummy is really biting",
            
            # UK expressions with aggravating factors
            "my tummy plays up when I eat",
            "I've got a dodgy belly when I'm stressed",
            "my stomach gives me gyp after meals",
            "my gut bothers me when I move",
            "my belly niggles when I lie down",
            "I've got a twinge in my tummy when I cough",
            "my stomach is not quite right when I'm hungry",
            "my gut is off colour when I'm tired",
            "my belly feels rough when I'm anxious",
            "my tummy is terrible when I'm worried",
            
            # UK expressions with associated symptoms
            "my tummy is playing up and I feel sick",
            "I've got a dodgy belly and I'm queasy",
            "my stomach is giving me gyp and I want to be sick",
            "my gut is bothering me and I feel nauseous",
            "my belly is niggling and I feel dizzy",
            "I've got a twinge in my tummy and I feel faint",
            "my stomach is not quite right and I feel weak",
            "my gut is off colour and I feel tired",
            "my belly feels rough and I feel unwell",
            "my tummy is terrible and I feel poorly",
            
            # Additional UK expressions with more variations
            "my tummy's been playing up something rotten",
            "I've got a right dodgy belly on me",
            "my stomach's giving me proper gyp",
            "there's something not right with my gut",
            "my tummy's been bothering me no end",
            "I've got a bit of bother with my belly area",
            "my stomach's been niggling away at me",
            "I've got a proper twinge in my tummy",
            "my belly's not quite the ticket",
            "I'm feeling a bit peaky in my stomach region",
            "my gut's gone all off colour",
            "I'm not feeling great in my tummy department",
            "my belly's been feeling really rough",
            "I've got a nasty pain in my stomach area",
            "my tummy's been absolutely terrible lately",
            "I feel proper rubbish in my belly",
            "my gut's been bloody awful today",
            "my stomach's not been brilliant recently",
            "I've got a horrible pain in my tummy region",
            "my belly's been pretty rubbish for ages",
            
            # UK expressions with more timing variations
            "my tummy's been playing up for donkey's years",
            "I've had a dodgy belly since the crack of dawn",
            "my stomach's been giving me gyp all day long",
            "my gut's been bothering me on and off for weeks",
            "my belly's been niggling for what seems like forever",
            "I've had a twinge in my tummy since yesterday morning",
            "my stomach's been not quite right for ages now",
            "I've been feeling a bit peaky in my belly for days",
            "my gut's been off colour since last weekend",
            "my tummy's been feeling rough all week long",
            "I've had a nasty pain in my belly for months",
            "my stomach's been absolutely terrible since Monday",
            "I've been feeling rubbish in my tummy for weeks",
            "my gut's been bloody awful since the weekend",
            "my belly's been not brilliant for quite a while",
            "I've had a horrible pain in my tummy since this afternoon",
            "my stomach's been pretty rubbish for months on end",
            
            # UK expressions with more severity variations
            "my tummy's a bit dodgy today",
            "I've got a really dodgy belly on me",
            "my stomach's quite niggling this morning",
            "my gut's really bothering me something chronic",
            "my belly's quite painful when I move",
            "I've got a really nasty pain in my tummy",
            "my stomach's quite terrible this evening",
            "my gut's really rubbish today",
            "my belly's quite awful when I eat",
            "my tummy's really not brilliant lately",
            "I've got a quite horrible pain in my belly",
            "my stomach's really pretty rubbish today",
            "my tummy's a bit of a nuisance",
            "I've got a proper dodgy belly",
            "my stomach's quite a bother",
            "my gut's really playing up",
            "my belly's quite a pain",
            "I've got a right nasty pain in my tummy",
            "my stomach's quite terrible",
            "my gut's really not good",
            
            # UK expressions with more character variations
            "my tummy's a bit achy today",
            "I've got a sharp pain in my dodgy belly",
            "my stomach's quite crampy this morning",
            "my gut's really burning when I eat",
            "my belly's quite stabbing this evening",
            "I've got a really gnawing pain in my tummy",
            "my stomach's quite throbbing today",
            "my gut's really shooting when I move",
            "my belly's quite stinging this afternoon",
            "my tummy's really biting when I cough",
            "my stomach's a bit sore",
            "I've got a dull ache in my dodgy belly",
            "my gut's quite tender",
            "my belly's really sensitive",
            "my tummy's quite uncomfortable",
            "I've got a nagging pain in my belly",
            "my stomach's quite persistent",
            "my gut's really annoying",
            "my belly's quite bothersome",
            "my tummy's really irritating",
            
            # UK expressions with more aggravating factors
            "my tummy plays up when I eat spicy food",
            "I've got a dodgy belly when I'm stressed out",
            "my stomach gives me gyp after big meals",
            "my gut bothers me when I move around",
            "my belly niggles when I lie down flat",
            "I've got a twinge in my tummy when I cough hard",
            "my stomach's not quite right when I'm hungry",
            "my gut's off colour when I'm tired out",
            "my belly feels rough when I'm anxious",
            "my tummy's terrible when I'm worried sick",
            "my stomach plays up when I drink coffee",
            "I've got a dodgy belly when I'm nervous",
            "my gut gives me gyp when I eat too much",
            "my belly bothers me when I bend over",
            "my tummy niggles when I'm stressed",
            "I've got a twinge when I take deep breaths",
            "my stomach's not right when I'm upset",
            "my gut's off when I'm run down",
            "my belly feels rough when I'm tired",
            "my tummy's awful when I'm worried",
            
            # UK expressions with more associated symptoms
            "my tummy's playing up and I feel sick as a dog",
            "I've got a dodgy belly and I'm feeling queasy",
            "my stomach's giving me gyp and I want to be sick",
            "my gut's bothering me and I feel nauseous",
            "my belly's niggling and I feel dizzy",
            "I've got a twinge in my tummy and I feel faint",
            "my stomach's not quite right and I feel weak",
            "my gut's off colour and I feel tired out",
            "my belly feels rough and I feel unwell",
            "my tummy's terrible and I feel poorly",
            "my stomach's playing up and I feel sick",
            "I've got a dodgy belly and I'm feeling off",
            "my gut's giving me gyp and I feel nauseous",
            "my belly's bothering me and I feel dizzy",
            "my tummy's niggling and I feel faint",
            "I've got a twinge and I feel weak",
            "my stomach's not right and I feel tired",
            "my gut's off and I feel unwell",
            "my belly feels rough and I feel poorly",
            "my tummy's awful and I feel sick",
        ]
        test_cases.extend(uk_expressions)
        
        # Add reduced medical scenarios (minority of test cases)
        medical_scenarios = [
            "I have pain in my right upper quadrant that radiates to my right shoulder",
            "my left lower quadrant is tender and I have a fever",
            "there's a sharp pain in my epigastric region that goes through to my back",
            "I have cramping pain in my periumbilical area that moves around",
            "my right lower quadrant pain started around my belly button and moved",
            "I have burning pain in my retrosternal area that gets worse when I eat",
            "my suprapubic area is painful and I have urinary symptoms",
            "I have colicky pain in my flank that radiates to my groin",
            "my epigastric pain is worse when I\'m hungry and better when I eat",
            "I have diffuse abdominal pain with nausea and vomiting",
        ]
        test_cases.extend(medical_scenarios)
        
        # Remove duplicates and limit to 500 (UK language focused)
        test_cases = list(set(test_cases))
        if len(test_cases) > 500:
            test_cases = test_cases[:500]
        
        return test_cases
    
    # Generate test cases
    print(f"\n🔄 Generating comprehensive test cases...")
    test_cases = generate_comprehensive_test_cases()
    print(f"✅ Generated {len(test_cases)} test cases")
    
    # Run comprehensive testing
    print(f"\n🔍 Running comprehensive OLDCARTS normalization test...")
    print("=" * 70)
    
    start_time = time.time()
    results = {
        'total_tests': len(test_cases),
        'normalized_count': 0,
        'not_normalized_count': 0,
        'normalization_matches': {},
        'sample_results': [],
        'uk_expressions_results': [],
        'complex_combinations_results': [],
    }
    
    for i, test_input in enumerate(test_cases):
        normalized, matches = normalize_text(test_input)
        
        if matches:
            results['normalized_count'] += 1
            for variation, standard_term in matches:
                if standard_term not in results['normalization_matches']:
                    results['normalization_matches'][standard_term] = 0
                results['normalization_matches'][standard_term] += 1
        else:
            results['not_normalized_count'] += 1
        
        # Store sample results for analysis
        if i < 50:  # First 50 results
            results['sample_results'].append({
                'input': test_input,
                'normalized': normalized,
                'matches': matches,
                'normalized_count': len(matches)
            })
        
        # Store UK expression results
        if any(uk_term in test_input.lower() for uk_term in ['playing up', 'dodgy', 'gyp', 'niggling', 'twinge', 'peaky', 'rough', 'rubbish']):
            results['uk_expressions_results'].append({
                'input': test_input,
                'normalized': normalized,
                'matches': matches,
                'normalized_count': len(matches)
            })
        
        # Store complex combination results
        if len(test_input.split()) > 8:  # Complex sentences
            results['complex_combinations_results'].append({
                'input': test_input,
                'normalized': normalized,
                'matches': matches,
                'normalized_count': len(matches)
            })
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(test_cases)} test cases...")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Display results
    print(f"\n📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 70)
    print(f"Total test cases: {results['total_tests']}")
    print(f"Successfully normalized: {results['normalized_count']}")
    print(f"Not normalized: {results['not_normalized_count']}")
    print(f"Normalization rate: {(results['normalized_count'] / results['total_tests']) * 100:.1f}%")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Average time per test: {processing_time / results['total_tests']:.4f} seconds")
    
    # Top normalization categories
    print(f"\n🏆 TOP NORMALIZATION CATEGORIES:")
    sorted_matches = sorted(results['normalization_matches'].items(), key=lambda x: x[1], reverse=True)
    for i, (category, count) in enumerate(sorted_matches[:15], 1):
        print(f"   {i:2d}. {category}: {count} matches")
    
    # Sample results
    print(f"\n📋 SAMPLE RESULTS (First 10):")
    for i, result in enumerate(results['sample_results'][:10], 1):
        print(f"   {i:2d}. Input: '{result['input']}'")
        print(f"       Normalized: '{result['normalized']}'")
        print(f"       Matches: {result['normalized_count']}")
        if result['matches']:
            for variation, standard_term in result['matches']:
                print(f"          - '{variation}' → '{standard_term}'")
        print()
    
    # UK expressions results
    if results['uk_expressions_results']:
        print(f"\n🇬🇧 UK EXPRESSIONS RESULTS ({len(results['uk_expressions_results'])} found):")
        for i, result in enumerate(results['uk_expressions_results'][:5], 1):
            print(f"   {i}. Input: '{result['input']}'")
            print(f"      Normalized: '{result['normalized']}'")
            print(f"      Matches: {result['normalized_count']}")
            if result['matches']:
                for variation, standard_term in result['matches']:
                    print(f"         - '{variation}' → '{standard_term}'")
            print()
    
    # Complex combinations results
    if results['complex_combinations_results']:
        print(f"\n🔗 COMPLEX COMBINATIONS RESULTS ({len(results['complex_combinations_results'])} found):")
        for i, result in enumerate(results['complex_combinations_results'][:5], 1):
            print(f"   {i}. Input: '{result['input']}'")
            print(f"      Normalized: '{result['normalized']}'")
            print(f"      Matches: {result['normalized_count']}")
            if result['matches']:
                for variation, standard_term in result['matches']:
                    print(f"         - '{variation}' → '{standard_term}'")
            print()
    
    # Save detailed results
    output_file = "oldcarts_comprehensive_gi_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print(f"\n✅ Comprehensive OLDCARTS GI normalization test completed!")
    
    return True

if __name__ == "__main__":
    test_oldcarts_comprehensive_gi()
