"""
Step 1: Generate diverse training sentences for Piper TTS voice training.

Produces ~10,000 sentences covering the full phonetic range of English,
with emphasis on conversational AI assistant speech patterns (what Aura
actually says). Targets ~8-10 hours of audio at ~3-4 seconds per sentence.

Usage:
    python generate_sentences.py [--count 10000] [--output sentences.txt]
"""

import argparse
import random
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Sentence templates — diverse phonetic coverage + Aura-style speech
# ---------------------------------------------------------------------------

# Conversational AI assistant responses (what Aura actually says)
ASSISTANT_RESPONSES = [
    "Good morning! How can I help you today?",
    "That's a great question. Let me think about that.",
    "I'd be happy to help you with that.",
    "Based on what I know, here's what I can tell you.",
    "Sure, I can help with that right away.",
    "Let me look into that for you.",
    "That's really interesting. Tell me more.",
    "I understand. Here's what I'd suggest.",
    "Of course! Here's what you need to know.",
    "I'm not entirely sure about that, but I'll do my best.",
    "Welcome back! It's nice to see you again.",
    "I have some updates that might interest you.",
    "Would you like me to go into more detail?",
    "Is there anything else I can help you with?",
    "That makes perfect sense.",
    "I appreciate you sharing that with me.",
    "Here's a quick summary for you.",
    "Let me break that down step by step.",
    "Great choice! Here's how to get started.",
    "I completely understand your concern.",
    "You raise a really good point.",
    "That's absolutely right.",
    "I'd recommend starting with the basics first.",
    "Here's what I found for you.",
    "I think you'll find this helpful.",
    "No problem at all. Let me explain.",
    "I see what you mean. Let me clarify.",
    "That's a common question, and the answer is straightforward.",
    "I want to make sure I'm understanding you correctly.",
    "Good evening. I hope you had a productive day.",
    "I have a brief prepared for you about recent developments.",
    "The weather today looks clear with mild temperatures.",
    "Your schedule for tomorrow includes two meetings.",
    "I noticed something that might be worth your attention.",
    "Based on the latest data, here are the key highlights.",
    "I'd suggest taking a closer look at the details.",
    "Everything looks good on my end.",
    "I'll keep monitoring that for you.",
    "Consider this a gentle reminder about your upcoming deadline.",
    "I'm ready whenever you are.",
]

# General knowledge / informational
INFORMATIONAL = [
    "The capital of France is Paris, known for its iconic Eiffel Tower.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The human heart beats approximately seventy times per minute.",
    "Light travels at roughly three hundred thousand kilometers per second.",
    "The Great Wall of China stretches over thirteen thousand miles.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "Shakespeare wrote thirty seven plays during his lifetime.",
    "DNA stands for deoxyribonucleic acid and carries genetic information.",
    "The moon orbits the Earth approximately every twenty seven days.",
    "Mount Everest stands at eight thousand eight hundred forty eight meters.",
    "The Amazon River is the second longest river in the world.",
    "Gravity on Mars is about thirty eight percent of Earth's gravity.",
    "The speed of sound in air is roughly three hundred forty three meters per second.",
    "A marathon is exactly twenty six point two miles long.",
    "The periodic table contains one hundred eighteen known elements.",
    "Venus is the hottest planet in our solar system.",
    "Honey never spoils if stored properly in a sealed container.",
    "The Sahara Desert covers most of northern Africa.",
    "Octopuses have three hearts and blue blood.",
]

# Recipes and instructions (Aura gets asked these a lot)
RECIPES_INSTRUCTIONS = [
    "First, preheat your oven to three hundred fifty degrees Fahrenheit.",
    "Mix two cups of flour with one teaspoon of baking powder.",
    "Add three tablespoons of olive oil to a heated pan.",
    "Stir the mixture until it reaches a smooth consistency.",
    "Let the dough rest for about thirty minutes at room temperature.",
    "Season with salt, pepper, and a pinch of paprika.",
    "Cook on medium heat for approximately fifteen minutes.",
    "Bring four cups of water to a rolling boil.",
    "Fold the egg whites gently into the batter.",
    "Serve immediately while still warm for the best flavor.",
    "Chop the onions finely and sauté until translucent.",
    "Marinate the chicken for at least two hours in the refrigerator.",
    "Reduce the sauce by half over low heat, stirring occasionally.",
    "Garnish with fresh basil leaves and a drizzle of balsamic vinegar.",
    "The internal temperature should reach one hundred sixty five degrees Fahrenheit.",
    "Whisk together the eggs, sugar, and vanilla extract.",
    "Roll out the pastry dough to about one quarter inch thickness.",
    "Toast the bread until golden brown on both sides.",
    "Dissolve the yeast in warm water and let it sit for five minutes.",
    "Drain the pasta and toss with butter and parmesan cheese.",
]

# Numbers, dates, times (important for TTS to get right)
NUMBERS_DATES = [
    "Your appointment is scheduled for March twenty first at two thirty.",
    "The total comes to forty seven dollars and sixty three cents.",
    "The meeting starts at nine fifteen in the morning.",
    "There are approximately seven point eight billion people on Earth.",
    "The project deadline is January fifteenth, twenty twenty seven.",
    "That's about three and a half hours from now.",
    "The temperature today is seventy two degrees Fahrenheit.",
    "Your flight departs at six forty five in the evening.",
    "The package weighs about twelve point five pounds.",
    "It's currently four seventeen in the afternoon.",
    "The store closes at eight o'clock tonight.",
    "That happened on November third, nineteen eighty nine.",
    "The distance is roughly two hundred and fifty miles.",
    "Your reservation is for a party of six at seven thirty.",
    "The building has twenty three floors above ground.",
    "Gas prices are around three dollars and forty nine cents per gallon.",
    "The population grew by two point three percent last year.",
    "We need about fifteen to twenty minutes to finish.",
    "The concert starts at half past eight this Saturday.",
    "That's the third time this week it's happened.",
]

# Emotional / empathetic responses
EMOTIONAL = [
    "I'm really sorry to hear that. That must be difficult.",
    "That's wonderful news! Congratulations!",
    "I can understand why you'd feel that way.",
    "Don't worry, these things happen to everyone.",
    "I'm so happy that worked out for you.",
    "Take your time. There's no rush at all.",
    "That sounds like a really challenging situation.",
    "You should be proud of what you've accomplished.",
    "I'm here if you need to talk about it.",
    "What a lovely thing to say. Thank you.",
    "I believe you can absolutely do this.",
    "That takes a lot of courage. I respect that.",
    "Sometimes the best thing to do is take a step back.",
    "I think you're being too hard on yourself.",
    "That's perfectly normal. Don't worry about it.",
    "How exciting! Tell me all about it.",
    "I appreciate your patience with this.",
    "You've made remarkable progress already.",
    "It's okay to not have all the answers right away.",
    "That sounds like a fantastic opportunity.",
]

# Phonetically rich sentences (Harvard sentences style — cover all phonemes)
PHONETIC_RICH = [
    "The quick brown fox jumps over the lazy dog near the bridge.",
    "She sells seashells by the seashore every sunny Saturday.",
    "Peter Piper picked a peck of pickled peppers in the garden.",
    "How much wood would a woodchuck chuck if it could?",
    "The thick fog crept through the narrow valley at dawn.",
    "A gentle breeze rustled the orange leaves on the old oak tree.",
    "The judge gave a very specific and thorough explanation.",
    "Twelve musicians played jazz rhythms throughout the evening.",
    "The treasure was buried beneath the ancient church tower.",
    "A bright yellow butterfly landed on the purple lavender bush.",
    "The chef prepared an exquisite dish with fresh ingredients.",
    "Children laughed and played in the warm afternoon sunshine.",
    "The mysterious stranger vanished into the thick morning mist.",
    "Several thousand spectators watched the thrilling championship game.",
    "The ancient manuscript contained valuable historical knowledge.",
    "A magnificent rainbow appeared after the thunderstorm passed.",
    "The professor explained the complex theory with simple examples.",
    "Fresh strawberries and whipped cream make a delightful dessert.",
    "The orchestra performed a breathtaking symphony in the grand hall.",
    "Winding mountain roads offered spectacular views of the coastline.",
    "The photographer captured a stunning image of the northern lights.",
    "Heavy rain disrupted travel plans across the entire region.",
    "The volunteers organized a successful fundraising event downtown.",
    "An unexpected visitor arrived just before midnight on Thursday.",
    "The librarian recommended several fascinating novels for the summer.",
    "Dozens of colorful kites soared high above the sandy beach.",
    "The mechanic diagnosed the problem with remarkable efficiency.",
    "Beautiful architecture and rich history define this charming city.",
    "The young scientist discovered a breakthrough in renewable energy.",
    "Warm cinnamon rolls fresh from the oven filled the kitchen with fragrance.",
]

# Questions (Aura asks these)
QUESTIONS = [
    "Would you like me to elaborate on that?",
    "Is there anything specific you'd like to know more about?",
    "Shall I continue with the next topic?",
    "Does that answer your question?",
    "Would you prefer a shorter or more detailed explanation?",
    "Can I help you with anything else today?",
    "Do you have any preferences for how I should proceed?",
    "Would you like me to save that information for later?",
    "Should I set a reminder for that?",
    "Are you comfortable with that approach?",
    "What would you like to focus on first?",
    "Do you need me to repeat any of that?",
    "How does that sound to you?",
    "Would tomorrow morning work better for you?",
    "Is this a good time, or should I come back to this later?",
]

# Short utterances (greetings, acknowledgements — must sound natural)
SHORT = [
    "Hello!",
    "Good morning!",
    "Good afternoon!",
    "Good evening!",
    "Of course!",
    "Absolutely!",
    "Welcome back!",
    "Certainly!",
    "Right away!",
    "No problem!",
    "Got it!",
    "Sure thing!",
    "My pleasure!",
    "You're welcome!",
    "Take care!",
    "Goodbye!",
    "See you later!",
    "That's great!",
    "Wonderful!",
    "Perfect!",
    "Understood!",
    "I see!",
    "Interesting!",
    "Exactly!",
    "Indeed!",
    "Glad to hear it!",
    "Not at all!",
    "With pleasure!",
    "Alright!",
    "Sounds good!",
]

# Longer, flowing passages (tests prosody and breath control)
LONG_PASSAGES = [
    "I've been analyzing the data from the past week, and there are a few trends worth noting. Overall engagement is up, but there's a slight dip in afternoon activity that might be worth investigating further.",
    "The best approach would be to start with a clear plan, break it into manageable steps, and tackle each one systematically. Don't try to do everything at once. Focus on the highest priority items first.",
    "Based on my research, there are three main options available to you. Each has its own advantages and trade-offs, so the right choice really depends on your specific priorities and timeline.",
    "I noticed that the system has been running smoothly for the past several days. All services are healthy, memory usage is within normal ranges, and there haven't been any reported issues.",
    "That's a complex topic with many different perspectives. The short answer is that it depends on several factors, including your budget, your timeline, and what you're ultimately trying to achieve.",
    "Good morning! I've compiled a summary of the key developments overnight. There are two items that require your attention, and three that are informational only. Shall I walk you through them?",
    "The recipe calls for fresh ingredients that you can find at most grocery stores. The key is to not overcook the vegetables, as they should retain a slight crunch for the best texture and flavor.",
    "I recommend scheduling that for early next week, which gives you enough time to prepare without feeling rushed. Tuesday or Wednesday would work well based on your current calendar.",
    "The weather forecast shows clear skies for the rest of the week, with temperatures gradually warming up toward the weekend. Saturday looks particularly nice for outdoor activities.",
    "Let me give you a quick overview of what happened today. Revenue was up three percent compared to yesterday, customer satisfaction scores remained steady, and we received positive feedback on the new feature launch.",
]

# Technology and business (common Aura topics)
TECH_BUSINESS = [
    "The software update includes several performance improvements and bug fixes.",
    "Cloud computing has transformed how businesses manage their infrastructure.",
    "Artificial intelligence is being used in healthcare, finance, and education.",
    "The quarterly report shows strong growth in the enterprise segment.",
    "Cybersecurity remains a top priority for organizations of all sizes.",
    "The new feature will be available to all users starting next month.",
    "Data analytics can help identify patterns that aren't immediately obvious.",
    "Remote work has become a permanent option for many companies.",
    "The startup raised twenty million dollars in their latest funding round.",
    "Blockchain technology has applications beyond cryptocurrency.",
    "User experience design plays a crucial role in product adoption.",
    "The API integration was completed ahead of schedule.",
    "Machine learning models require large amounts of quality training data.",
    "The dashboard provides real-time visibility into key performance metrics.",
    "Digital transformation is reshaping traditional industries worldwide.",
]

# Sentences with tricky pronunciations
TRICKY_PRONUNCIATION = [
    "The colonel was thoroughly familiar with the archipelago's unique characteristics.",
    "The entrepreneur's innovative approach to the pharmaceutical industry was remarkable.",
    "Wednesday's itinerary includes a visit to the Renaissance exhibition.",
    "The lieutenant acknowledged the significance of the archaeological discovery.",
    "The recipe requires Worcestershire sauce and quinoa as key ingredients.",
    "The chauffeur drove through the picturesque Mediterranean countryside.",
    "Their thorough analysis of the algorithm yielded surprising results.",
    "The enthusiastic mathematician explained the Fibonacci sequence beautifully.",
    "The genre of the documentary was difficult to categorize precisely.",
    "She particularly enjoyed the hors d'oeuvres at the elegant soirée.",
]


ALL_CATEGORIES = [
    (ASSISTANT_RESPONSES, 3.0),   # Heavy weight — this is what Aura says
    (INFORMATIONAL, 1.5),
    (RECIPES_INSTRUCTIONS, 1.5),
    (NUMBERS_DATES, 1.5),
    (EMOTIONAL, 2.0),
    (PHONETIC_RICH, 2.0),
    (QUESTIONS, 2.0),
    (SHORT, 1.5),
    (LONG_PASSAGES, 1.0),
    (TECH_BUSINESS, 1.0),
    (TRICKY_PRONUNCIATION, 1.0),
]


def generate_sentences(count: int) -> list[str]:
    """Generate a weighted random selection of sentences."""
    pool: list[tuple[str, float]] = []
    for category, weight in ALL_CATEGORIES:
        for sent in category:
            pool.append((sent, weight))

    sentences = []
    seen = set()

    # First pass: include every unique sentence at least once
    all_unique = [s for s, _ in pool]
    random.shuffle(all_unique)
    for s in all_unique:
        if s not in seen:
            sentences.append(s)
            seen.add(s)

    # Second pass: weighted sampling to reach target count
    items, weights = zip(*pool)
    while len(sentences) < count:
        # Pick from weighted pool, allow repeats for common patterns
        chosen = random.choices(items, weights=weights, k=min(100, count - len(sentences)))
        for s in chosen:
            if len(sentences) >= count:
                break
            sentences.append(s)

    random.shuffle(sentences)
    return sentences[:count]


def main():
    parser = argparse.ArgumentParser(description="Generate training sentences for Piper TTS")
    parser.add_argument("--count", type=int, default=10000,
                        help="Number of sentences to generate (default: 10000)")
    parser.add_argument("--output", type=str, default="sentences.txt",
                        help="Output file path (default: sentences.txt)")
    args = parser.parse_args()

    sentences = generate_sentences(args.count)

    out = Path(args.output)
    out.write_text("\n".join(sentences) + "\n", encoding="utf-8")

    # Stats
    unique = len(set(sentences))
    total_chars = sum(len(s) for s in sentences)
    avg_chars = total_chars / len(sentences)
    est_hours = (len(sentences) * 3.5) / 3600  # ~3.5s avg per sentence

    print(f"Generated {len(sentences)} sentences ({unique} unique)")
    print(f"Total characters: {total_chars:,}")
    print(f"Average length: {avg_chars:.0f} chars")
    print(f"Estimated audio duration: {est_hours:.1f} hours")
    print(f"Written to: {out}")


if __name__ == "__main__":
    main()
