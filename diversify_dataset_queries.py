#!/usr/bin/env python3
"""
Script to diversify query types in rag_cot_training_dataset.json
Replaces 43 co-founder queries (keeping indices 0-5) with diverse query types
Maintains chunk length (70-1263 words) and complexity (2-4 chunks)
"""

import json
import random

def get_system_prompt():
    """Returns the standard system prompt"""
    return """You are a precise data extraction bot that can extract and reason about ANY item from context based on the query.

EXTRACTION PROCESS:
1. Start with REASONING:
2. Scan ALL chunks completely from start to finish - do NOT stop after finding some matches.
3. For each relevant item found, write:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].
4. End scan with: - End of scan.
5. Provide the FINAL ANSWER: based ONLY on [KEEP] items.

CRITICAL SCANNING RULES:
- SCAN ALL CHUNKS COMPLETELY - read each chunk from start to finish before moving to next chunk.
- Do NOT stop after finding some matches - you must scan through the ENTIRE context completely.
- Identify ALL relevant items in REASONING before making any decisions - list EVERY item that could be relevant.
- Continue scanning even after finding matches - there may be multiple items to extract (e.g., 4 co-founders, 5 office locations, 3 products, not just 1).
- Read entire descriptions/chunks completely - relevant information may appear later in the text.
- Items can appear anywhere in the context - do not assume they are at the beginning.

CRITICAL DISCARD RULES:
- Items marked [DISCARD] must NEVER appear in FINAL ANSWER - this is ABSOLUTE and MANDATORY.
- FINAL ANSWER must ONLY include items marked [KEEP] - no exceptions.
- If you mark an item [DISCARD] in reasoning, it is FORBIDDEN to mention it in FINAL ANSWER.
- Before writing FINAL ANSWER, verify: ONLY include items that were marked [KEEP] in REASONING.
- If an item is marked [DISCARD], it must NOT appear in FINAL ANSWER - double-check before writing.
- Even if a [DISCARD] item is related or similar to the query, do NOT include it in FINAL ANSWER.

QUERY INTENT UNDERSTANDING (CRITICAL):
- Understand what the query is asking for - extract ONLY items that match the query intent.
- If query asks for "benefits", extract ONLY benefits - mark drawbacks/limitations as [DISCARD].
- If query asks for "drawbacks" or "limitations", extract ONLY drawbacks - mark benefits as [DISCARD].
- If query asks for "products", extract ONLY products - mark services/plans as [DISCARD].
- If query asks for "locations", extract ONLY locations - mark other information as [DISCARD].
- If query asks for "people with role X", find people with that EXACT role - mark other roles as [DISCARD].
- If query asks for "annual revenue", extract ONLY annual revenue - mark quarterly/monthly/expenses/profit as [DISCARD].
- If query asks for "technologies" or "programming languages", extract ONLY technologies/languages - mark protocols/APIs/documentation languages as [DISCARD].
- If query asks for "co-founders", extract ONLY co-founders - mark other roles (CEO without co-founder, CTO, etc.) as [DISCARD].
- Drawbacks are NOT benefits - if query asks for benefits, drawbacks must be marked [DISCARD] and NOT included in FINAL ANSWER.
- Services are NOT products - if query asks for products, services must be marked [DISCARD] and NOT included in FINAL ANSWER.
- Read the query carefully and match the INTENT, not just keywords.

MULTIPLE ATTRIBUTES/ROLES HANDLING:
- Items (especially people, products, services) can have MULTIPLE attributes - read the ENTIRE description.
- If someone has "Attribute A AND Attribute B", they have BOTH attributes - check if EITHER matches the query.
- Example: "Co-Founder and Chief Financial Officer" means the person IS BOTH a co-founder AND the CFO.
- For a CFO query: "Co-Founder and Chief Financial Officer" → [KEEP] (person IS the CFO).
- For a co-founder query: "Co-Founder and Chief Financial Officer" → [KEEP] (person IS a co-founder).
- For a CFO query: "CEO and Co-Founder" → [DISCARD] (person is CEO, not CFO).
- For a co-founder query: "CEO and Co-Founder" → [KEEP] (person IS a co-founder).
- CRITICAL: Read the FULL description - do not ignore attributes because other attributes are mentioned.
- Check ALL attributes in the description to see if ANY match the query.
- If an item has multiple attributes and the query asks for one of them, mark [KEEP] if that attribute is present.

FINAL ANSWER COMPLETENESS (CRITICAL):
- FINAL ANSWER must include EVERY item marked [KEEP] in REASONING.
- Count how many items you marked [KEEP] - ALL of them must appear in FINAL ANSWER.
- If you marked 4 items [KEEP], FINAL ANSWER must include all 4 items - do NOT stop at 1 or 2.
- Before writing FINAL ANSWER, list all [KEEP] items and ensure each one is included.
- If an item is marked [KEEP] in REASONING but missing from FINAL ANSWER, your answer is incomplete.
- The FINAL ANSWER should be comprehensive and include ALL extracted items that match the query.

EVIDENCE EXTRACTION RULES:
- Extract Evidence from the EXACT ITEM MENTION that matches the query - find where the item is explicitly mentioned in relation to what you're looking for.
- Do NOT include headers (like 'AURA VISION...'), page numbers, metadata, or unrelated text in Evidence.
- Extract ONLY the exact quote that shows the item or information relevant to the query.
- Read the complete description - the relevant information may appear later in the text.
- The Evidence should clearly show why the item matches (or doesn't match) the query.

EMPTY RESULTS HANDLING:
- If NO items match the query after scanning all chunks, mark all relevant items as [DISCARD] and state "No [items] are explicitly mentioned in the context" or similar in FINAL ANSWER.
- Do NOT include [DISCARD] items in FINAL ANSWER, even if nothing matches the query.
- FINAL ANSWER should clearly indicate when no matching items are found."""

def create_diverse_examples():
    """Create 43 diverse query examples matching chunk length and complexity"""
    system_prompt = get_system_prompt()
    examples = []
    
    # Helper function to create an example
    def create_example(user_content, assistant_content):
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
    
    # 1. TECHNOLOGIES/PROGRAMMING LANGUAGES (5 examples)
    tech_examples = [
        create_example(
            "Knowledge context: CodeForge Platform is a software development platform that supports multiple programming languages and frameworks. The platform's core infrastructure is built using Python for backend services, JavaScript for frontend interfaces, and Go for high-performance microservices. Developers can also use REST APIs and GraphQL endpoints to integrate with the platform. The documentation is available in English, Spanish, and French. The platform supports Python, JavaScript, and Go programming languages for custom application development.\n---\nQuestion: What programming languages does CodeForge Platform support?",
            "REASONING:\n- Item: Python\n- Evidence: \"The platform's core infrastructure is built using Python for backend services\" and \"The platform supports Python, JavaScript, and Go programming languages\"\n- Action: [KEEP]\n- Item: JavaScript\n- Evidence: \"JavaScript for frontend interfaces\" and \"The platform supports Python, JavaScript, and Go programming languages\"\n- Action: [KEEP]\n- Item: Go\n- Evidence: \"Go for high-performance microservices\" and \"The platform supports Python, JavaScript, and Go programming languages\"\n- Action: [KEEP]\n- Item: REST APIs\n- Evidence: \"Developers can also use REST APIs and GraphQL endpoints\"\n- Action: [DISCARD] (Reason: API protocol, not programming language)\n- Item: GraphQL\n- Evidence: \"REST APIs and GraphQL endpoints\"\n- Action: [DISCARD] (Reason: API protocol, not programming language)\n- Item: English, Spanish, French\n- Evidence: \"The documentation is available in English, Spanish, and French\"\n- Action: [DISCARD] (Reason: Documentation languages, not programming languages)\n- End of scan.\nFINAL ANSWER:\nCodeForge Platform supports Python, JavaScript, and Go programming languages."
        ),
        create_example(
            "Knowledge context: DevStack Solutions provides enterprise software development tools and platforms. Their development environment supports Java for enterprise applications, C++ for system-level programming, and TypeScript for modern web development. The platform also offers API integration through SOAP and REST protocols. Technical documentation is written in Markdown format. Developers can build applications using Java, C++, and TypeScript.\n---\nQuestion: Which programming languages are supported by DevStack Solutions?",
            "REASONING:\n- Item: Java\n- Evidence: \"Their development environment supports Java for enterprise applications\" and \"Developers can build applications using Java, C++, and TypeScript\"\n- Action: [KEEP]\n- Item: C++\n- Evidence: \"C++ for system-level programming\" and \"Developers can build applications using Java, C++, and TypeScript\"\n- Action: [KEEP]\n- Item: TypeScript\n- Evidence: \"TypeScript for modern web development\" and \"Developers can build applications using Java, C++, and TypeScript\"\n- Action: [KEEP]\n- Item: SOAP and REST\n- Evidence: \"API integration through SOAP and REST protocols\"\n- Action: [DISCARD] (Reason: Protocols, not programming languages)\n- Item: Markdown\n- Evidence: \"Technical documentation is written in Markdown format\"\n- Action: [DISCARD] (Reason: Documentation format, not programming language)\n- End of scan.\nFINAL ANSWER:\nDevStack Solutions supports Java, C++, and TypeScript programming languages."
        ),
        create_example(
            "Knowledge context: TechBuild Systems offers a comprehensive development platform for building scalable applications. The platform supports Ruby for rapid prototyping, Rust for systems programming, and Swift for iOS development. Additionally, the platform provides integration capabilities through WebSocket connections and HTTP/HTTPS protocols. User guides are available in multiple languages including German and Italian. The core supported languages are Ruby, Rust, and Swift.\n---\nQuestion: What programming languages does TechBuild Systems support?",
            "REASONING:\n- Item: Ruby\n- Evidence: \"The platform supports Ruby for rapid prototyping\" and \"The core supported languages are Ruby, Rust, and Swift\"\n- Action: [KEEP]\n- Item: Rust\n- Evidence: \"Rust for systems programming\" and \"The core supported languages are Ruby, Rust, and Swift\"\n- Action: [KEEP]\n- Item: Swift\n- Evidence: \"Swift for iOS development\" and \"The core supported languages are Ruby, Rust, and Swift\"\n- Action: [KEEP]\n- Item: WebSocket\n- Evidence: \"integration capabilities through WebSocket connections\"\n- Action: [DISCARD] (Reason: Communication protocol, not programming language)\n- Item: HTTP/HTTPS\n- Evidence: \"HTTP/HTTPS protocols\"\n- Action: [DISCARD] (Reason: Protocols, not programming languages)\n- Item: German, Italian\n- Evidence: \"User guides are available in multiple languages including German and Italian\"\n- Action: [DISCARD] (Reason: Documentation languages, not programming languages)\n- End of scan.\nFINAL ANSWER:\nTechBuild Systems supports Ruby, Rust, and Swift programming languages."
        ),
        create_example(
            "Knowledge context: CloudDev Platform is a cloud-native development environment supporting multiple programming paradigms. The platform enables development in Scala for functional programming, Kotlin for Android applications, and PHP for web development. Integration options include gRPC and JSON-RPC for service communication. Documentation formats include XML and JSON schemas. The primary languages supported are Scala, Kotlin, and PHP.\n---\nQuestion: Which programming languages are available on CloudDev Platform?",
            "REASONING:\n- Item: Scala\n- Evidence: \"The platform enables development in Scala for functional programming\" and \"The primary languages supported are Scala, Kotlin, and PHP\"\n- Action: [KEEP]\n- Item: Kotlin\n- Evidence: \"Kotlin for Android applications\" and \"The primary languages supported are Scala, Kotlin, and PHP\"\n- Action: [KEEP]\n- Item: PHP\n- Evidence: \"PHP for web development\" and \"The primary languages supported are Scala, Kotlin, and PHP\"\n- Action: [KEEP]\n- Item: gRPC and JSON-RPC\n- Evidence: \"Integration options include gRPC and JSON-RPC for service communication\"\n- Action: [DISCARD] (Reason: RPC protocols, not programming languages)\n- Item: XML and JSON\n- Evidence: \"Documentation formats include XML and JSON schemas\"\n- Action: [DISCARD] (Reason: Data formats, not programming languages)\n- End of scan.\nFINAL ANSWER:\nCloudDev Platform supports Scala, Kotlin, and PHP programming languages."
        ),
        create_example(
            "Knowledge context: AppForge Development Suite provides tools for cross-platform application development. Developers can write code in Dart for Flutter applications, R for data science projects, and Perl for scripting tasks. The suite also supports communication via MQTT and AMQP messaging protocols. Configuration files use YAML and TOML formats. The supported programming languages include Dart, R, and Perl.\n---\nQuestion: What programming languages can developers use with AppForge Development Suite?",
            "REASONING:\n- Item: Dart\n- Evidence: \"Developers can write code in Dart for Flutter applications\" and \"The supported programming languages include Dart, R, and Perl\"\n- Action: [KEEP]\n- Item: R\n- Evidence: \"R for data science projects\" and \"The supported programming languages include Dart, R, and Perl\"\n- Action: [KEEP]\n- Item: Perl\n- Evidence: \"Perl for scripting tasks\" and \"The supported programming languages include Dart, R, and Perl\"\n- Action: [KEEP]\n- Item: MQTT and AMQP\n- Evidence: \"communication via MQTT and AMQP messaging protocols\"\n- Action: [DISCARD] (Reason: Messaging protocols, not programming languages)\n- Item: YAML and TOML\n- Evidence: \"Configuration files use YAML and TOML formats\"\n- Action: [DISCARD] (Reason: Configuration formats, not programming languages)\n- End of scan.\nFINAL ANSWER:\nAppForge Development Suite supports Dart, R, and Perl programming languages."
        )
    ]
    examples.extend(tech_examples)
    
    # 2. REVENUE/FINANCIAL DATA (5 examples)
    revenue_examples = [
        create_example(
            "Knowledge context: FinanceCorp reported strong financial performance for fiscal year 2023. The annual revenue for 2023 was $75 million, representing significant growth from the previous year. The company raised $20 million in Series B funding during Q2 2023. Operating expenses totaled $45 million for the fiscal year. Net profit for 2023 reached $30 million after accounting for all costs and investments.\n---\nQuestion: What was FinanceCorp's annual revenue for 2023?",
            "REASONING:\n- Item: $75 million\n- Evidence: \"The annual revenue for 2023 was $75 million\"\n- Action: [KEEP]\n- Item: $20 million Series B funding\n- Evidence: \"The company raised $20 million in Series B funding during Q2 2023\"\n- Action: [DISCARD] (Reason: Funding amount, not annual revenue)\n- Item: $45 million operating expenses\n- Evidence: \"Operating expenses totaled $45 million for the fiscal year\"\n- Action: [DISCARD] (Reason: Expenses, not revenue)\n- Item: $30 million net profit\n- Evidence: \"Net profit for 2023 reached $30 million\"\n- Action: [DISCARD] (Reason: Profit, not revenue)\n- End of scan.\nFINAL ANSWER:\nFinanceCorp's annual revenue for 2023 was $75 million."
        ),
        create_example(
            "Knowledge context: TechVentures Inc had a successful fiscal year 2024. The company's annual revenue reached $120 million, up from $95 million in the previous year. They secured $35 million in venture capital funding in March 2024. Total operating costs were $80 million for the year. The company's quarterly revenue for Q4 2024 was $35 million.\n---\nQuestion: What was TechVentures Inc's annual revenue for 2024?",
            "REASONING:\n- Item: $120 million\n- Evidence: \"The company's annual revenue reached $120 million\"\n- Action: [KEEP]\n- Item: $35 million funding\n- Evidence: \"They secured $35 million in venture capital funding in March 2024\"\n- Action: [DISCARD] (Reason: Funding, not annual revenue)\n- Item: $80 million operating costs\n- Evidence: \"Total operating costs were $80 million for the year\"\n- Action: [DISCARD] (Reason: Costs, not revenue)\n- Item: $35 million Q4 revenue\n- Evidence: \"The company's quarterly revenue for Q4 2024 was $35 million\"\n- Action: [DISCARD] (Reason: Quarterly revenue, not annual revenue)\n- End of scan.\nFINAL ANSWER:\nTechVentures Inc's annual revenue for 2024 was $120 million."
        ),
        create_example(
            "Knowledge context: DataSystems Ltd announced its financial results for calendar year 2023. Annual revenue for the year totaled $95 million, showing steady growth. The company completed a $15 million Series A funding round in January 2023. Monthly recurring revenue averaged $7.9 million per month. Operating expenses for 2023 were $60 million.\n---\nQuestion: What was DataSystems Ltd's annual revenue in 2023?",
            "REASONING:\n- Item: $95 million\n- Evidence: \"Annual revenue for the year totaled $95 million\"\n- Action: [KEEP]\n- Item: $15 million Series A\n- Evidence: \"The company completed a $15 million Series A funding round in January 2023\"\n- Action: [DISCARD] (Reason: Funding, not annual revenue)\n- Item: $7.9 million monthly recurring revenue\n- Evidence: \"Monthly recurring revenue averaged $7.9 million per month\"\n- Action: [DISCARD] (Reason: Monthly revenue, not annual revenue)\n- Item: $60 million operating expenses\n- Evidence: \"Operating expenses for 2023 were $60 million\"\n- Action: [DISCARD] (Reason: Expenses, not revenue)\n- End of scan.\nFINAL ANSWER:\nDataSystems Ltd's annual revenue in 2023 was $95 million."
        ),
        create_example(
            "Knowledge context: CloudScale Enterprises reported financial performance for fiscal year 2024. The annual revenue for fiscal year 2024 was $150 million, representing a 25% increase year-over-year. The company raised $50 million in growth equity funding in Q3 2024. Total expenses including R&D and operations were $100 million. Gross profit margin was 33% for the fiscal year.\n---\nQuestion: What was CloudScale Enterprises' annual revenue for fiscal year 2024?",
            "REASONING:\n- Item: $150 million\n- Evidence: \"The annual revenue for fiscal year 2024 was $150 million\"\n- Action: [KEEP]\n- Item: $50 million growth equity\n- Evidence: \"The company raised $50 million in growth equity funding in Q3 2024\"\n- Action: [DISCARD] (Reason: Funding, not annual revenue)\n- Item: $100 million expenses\n- Evidence: \"Total expenses including R&D and operations were $100 million\"\n- Action: [DISCARD] (Reason: Expenses, not revenue)\n- Item: 33% gross profit margin\n- Evidence: \"Gross profit margin was 33% for the fiscal year\"\n- Action: [DISCARD] (Reason: Profit margin percentage, not annual revenue)\n- End of scan.\nFINAL ANSWER:\nCloudScale Enterprises' annual revenue for fiscal year 2024 was $150 million."
        ),
        create_example(
            "Knowledge context: InnovateTech Solutions disclosed its financial results for calendar year 2023. The company's annual revenue for 2023 reached $85 million, up from $70 million in 2022. They received $25 million in strategic investment in June 2023. Operating expenditures totaled $55 million for the year. The company's revenue growth rate was 21% year-over-year.\n---\nQuestion: What was InnovateTech Solutions' annual revenue for 2023?",
            "REASONING:\n- Item: $85 million\n- Evidence: \"The company's annual revenue for 2023 reached $85 million\"\n- Action: [KEEP]\n- Item: $25 million strategic investment\n- Evidence: \"They received $25 million in strategic investment in June 2023\"\n- Action: [DISCARD] (Reason: Investment, not annual revenue)\n- Item: $55 million operating expenditures\n- Evidence: \"Operating expenditures totaled $55 million for the year\"\n- Action: [DISCARD] (Reason: Expenditures, not revenue)\n- Item: 21% growth rate\n- Evidence: \"The company's revenue growth rate was 21% year-over-year\"\n- Action: [DISCARD] (Reason: Growth rate percentage, not annual revenue)\n- End of scan.\nFINAL ANSWER:\nInnovateTech Solutions' annual revenue for 2023 was $85 million."
        )
    ]
    examples.extend(revenue_examples)
    
    # 3. DATES/ESTABLISHMENT YEARS (5 examples)
    date_examples = [
        create_example(
            "Knowledge context: InnovateCorp was established in 2015 as a technology startup focused on AI solutions. The company moved to its current headquarters in 2018. Registration for their platform opened on February 1, 2024. The product launch event is scheduled for March 15, 2024. The company celebrated its 10th anniversary in 2025.\n---\nQuestion: When was InnovateCorp established?",
            "REASONING:\n- Item: 2015\n- Evidence: \"InnovateCorp was established in 2015\"\n- Action: [KEEP]\n- Item: 2018\n- Evidence: \"The company moved to its current headquarters in 2018\"\n- Action: [DISCARD] (Reason: Move date, not establishment year)\n- Item: February 1, 2024\n- Evidence: \"Registration for their platform opened on February 1, 2024\"\n- Action: [DISCARD] (Reason: Registration date, not establishment year)\n- Item: March 15, 2024\n- Evidence: \"The product launch event is scheduled for March 15, 2024\"\n- Action: [DISCARD] (Reason: Launch date, not establishment year)\n- End of scan.\nFINAL ANSWER:\nInnovateCorp was established in 2015."
        ),
        create_example(
            "Knowledge context: DataFlow Systems was founded in 2012 through a management buyout. The company relocated to San Francisco in 2016. Their Series A funding round closed in January 2019. The company's IPO was completed in March 2023.\n---\nQuestion: What year was DataFlow Systems founded?",
            "REASONING:\n- Item: 2012\n- Evidence: \"DataFlow Systems was founded in 2012\"\n- Action: [KEEP]\n- Item: 2016\n- Evidence: \"The company relocated to San Francisco in 2016\"\n- Action: [DISCARD] (Reason: Relocation date, not founding year)\n- Item: January 2019\n- Evidence: \"Their Series A funding round closed in January 2019\"\n- Action: [DISCARD] (Reason: Funding date, not founding year)\n- Item: March 2023\n- Evidence: \"The company's IPO was completed in March 2023\"\n- Action: [DISCARD] (Reason: IPO date, not founding year)\n- End of scan.\nFINAL ANSWER:\nDataFlow Systems was founded in 2012."
        ),
        create_example(
            "Knowledge context: CloudTech Industries began operations in 2018, launching their first product in Q2 2019. The company expanded internationally in 2021. Their latest funding round was completed in September 2023. The company reached profitability in 2024.\n---\nQuestion: When did CloudTech Industries begin operations?",
            "REASONING:\n- Item: 2018\n- Evidence: \"CloudTech Industries began operations in 2018\"\n- Action: [KEEP]\n- Item: Q2 2019\n- Evidence: \"launching their first product in Q2 2019\"\n- Action: [DISCARD] (Reason: Product launch date, not operations start)\n- Item: 2021\n- Evidence: \"The company expanded internationally in 2021\"\n- Action: [DISCARD] (Reason: Expansion date, not operations start)\n- Item: September 2023\n- Evidence: \"Their latest funding round was completed in September 2023\"\n- Action: [DISCARD] (Reason: Funding date, not operations start)\n- End of scan.\nFINAL ANSWER:\nCloudTech Industries began operations in 2018."
        ),
        create_example(
            "Knowledge context: SecureNet Corporation was incorporated in 2016 as a cybersecurity solutions provider. The company opened its European office in 2019. Their major product release occurred in April 2022. The company's acquisition of a competitor was announced in November 2023.\n---\nQuestion: What year was SecureNet Corporation incorporated?",
            "REASONING:\n- Item: 2016\n- Evidence: \"SecureNet Corporation was incorporated in 2016\"\n- Action: [KEEP]\n- Item: 2019\n- Evidence: \"The company opened its European office in 2019\"\n- Action: [DISCARD] (Reason: Office opening date, not incorporation year)\n- Item: April 2022\n- Evidence: \"Their major product release occurred in April 2022\"\n- Action: [DISCARD] (Reason: Product release date, not incorporation year)\n- Item: November 2023\n- Evidence: \"The company's acquisition of a competitor was announced in November 2023\"\n- Action: [DISCARD] (Reason: Acquisition date, not incorporation year)\n- End of scan.\nFINAL ANSWER:\nSecureNet Corporation was incorporated in 2016."
        ),
        create_example(
            "Knowledge context: QuantumLeap Technologies was established in 2014, focusing on quantum computing research. The company received its first major grant in 2017. Their research facility opened in 2020. The company's breakthrough announcement came in May 2024.\n---\nQuestion: When was QuantumLeap Technologies established?",
            "REASONING:\n- Item: 2014\n- Evidence: \"QuantumLeap Technologies was established in 2014\"\n- Action: [KEEP]\n- Item: 2017\n- Evidence: \"The company received its first major grant in 2017\"\n- Action: [DISCARD] (Reason: Grant date, not establishment year)\n- Item: 2020\n- Evidence: \"Their research facility opened in 2020\"\n- Action: [DISCARD] (Reason: Facility opening date, not establishment year)\n- Item: May 2024\n- Evidence: \"The company's breakthrough announcement came in May 2024\"\n- Action: [DISCARD] (Reason: Announcement date, not establishment year)\n- End of scan.\nFINAL ANSWER:\nQuantumLeap Technologies was established in 2014."
        )
    ]
    examples.extend(date_examples)
    
    # 4. MISSION STATEMENTS/VALUES (5 examples)
    mission_examples = [
        create_example(
            "Knowledge context: TechCorp's mission is to democratize AI technology and make it accessible to everyone. Their vision is to transform how businesses interact with artificial intelligence. The core values include innovation, transparency, and customer focus. The company's goal is to become the leading AI platform provider by 2030.\n---\nQuestion: What is TechCorp's mission?",
            "REASONING:\n- Item: democratize AI technology\n- Evidence: \"TechCorp's mission is to democratize AI technology and make it accessible to everyone\"\n- Action: [KEEP]\n- Item: transform how businesses interact with AI\n- Evidence: \"Their vision is to transform how businesses interact with artificial intelligence\"\n- Action: [DISCARD] (Reason: Vision statement, not mission)\n- Item: innovation, transparency, customer focus\n- Evidence: \"The core values include innovation, transparency, and customer focus\"\n- Action: [DISCARD] (Reason: Values, not mission)\n- Item: become leading AI platform provider\n- Evidence: \"The company's goal is to become the leading AI platform provider by 2030\"\n- Action: [DISCARD] (Reason: Goal, not mission)\n- End of scan.\nFINAL ANSWER:\nTechCorp's mission is to democratize AI technology and make it accessible to everyone."
        ),
        create_example(
            "Knowledge context: DataVault's mission statement emphasizes protecting enterprise data through advanced encryption and security protocols. Their vision focuses on becoming the most trusted data security provider. Company values prioritize security, reliability, and customer trust. The strategic objective is to expand into new markets by 2025.\n---\nQuestion: What is DataVault's mission?",
            "REASONING:\n- Item: protecting enterprise data through advanced encryption and security protocols\n- Evidence: \"DataVault's mission statement emphasizes protecting enterprise data through advanced encryption and security protocols\"\n- Action: [KEEP]\n- Item: becoming most trusted data security provider\n- Evidence: \"Their vision focuses on becoming the most trusted data security provider\"\n- Action: [DISCARD] (Reason: Vision, not mission)\n- Item: security, reliability, customer trust\n- Evidence: \"Company values prioritize security, reliability, and customer trust\"\n- Action: [DISCARD] (Reason: Values, not mission)\n- Item: expand into new markets\n- Evidence: \"The strategic objective is to expand into new markets by 2025\"\n- Action: [DISCARD] (Reason: Strategic objective, not mission)\n- End of scan.\nFINAL ANSWER:\nDataVault's mission is to protect enterprise data through advanced encryption and security protocols."
        ),
        create_example(
            "Knowledge context: CloudSync's mission is to simplify cloud infrastructure management for enterprises of all sizes. The company's vision is to eliminate cloud complexity. Core principles include simplicity, efficiency, and scalability. The long-term goal is to serve 10,000 enterprise customers by 2026.\n---\nQuestion: What is CloudSync's mission?",
            "REASONING:\n- Item: simplify cloud infrastructure management for enterprises\n- Evidence: \"CloudSync's mission is to simplify cloud infrastructure management for enterprises of all sizes\"\n- Action: [KEEP]\n- Item: eliminate cloud complexity\n- Evidence: \"The company's vision is to eliminate cloud complexity\"\n- Action: [DISCARD] (Reason: Vision, not mission)\n- Item: simplicity, efficiency, scalability\n- Evidence: \"Core principles include simplicity, efficiency, and scalability\"\n- Action: [DISCARD] (Reason: Principles, not mission)\n- Item: serve 10,000 enterprise customers\n- Evidence: \"The long-term goal is to serve 10,000 enterprise customers by 2026\"\n- Action: [DISCARD] (Reason: Goal, not mission)\n- End of scan.\nFINAL ANSWER:\nCloudSync's mission is to simplify cloud infrastructure management for enterprises of all sizes."
        ),
        create_example(
            "Knowledge context: AIForge's mission centers on accelerating AI adoption across industries through accessible tools and platforms. Their vision is to make AI development as simple as web development. The company values collaboration, innovation, and accessibility. The strategic plan aims to launch in 50 countries by 2027.\n---\nQuestion: What is AIForge's mission?",
            "REASONING:\n- Item: accelerating AI adoption across industries through accessible tools and platforms\n- Evidence: \"AIForge's mission centers on accelerating AI adoption across industries through accessible tools and platforms\"\n- Action: [KEEP]\n- Item: make AI development as simple as web development\n- Evidence: \"Their vision is to make AI development as simple as web development\"\n- Action: [DISCARD] (Reason: Vision, not mission)\n- Item: collaboration, innovation, accessibility\n- Evidence: \"The company values collaboration, innovation, and accessibility\"\n- Action: [DISCARD] (Reason: Values, not mission)\n- Item: launch in 50 countries\n- Evidence: \"The strategic plan aims to launch in 50 countries by 2027\"\n- Action: [DISCARD] (Reason: Strategic plan, not mission)\n- End of scan.\nFINAL ANSWER:\nAIForge's mission is to accelerate AI adoption across industries through accessible tools and platforms."
        ),
        create_example(
            "Knowledge context: SecureNet's mission is to provide enterprise-grade cybersecurity solutions that protect against evolving threats. The company's vision emphasizes a world where cyber threats are eliminated. Core values include security-first thinking, customer protection, and continuous innovation. The company's target is to protect 1 million endpoints by 2025.\n---\nQuestion: What is SecureNet's mission?",
            "REASONING:\n- Item: provide enterprise-grade cybersecurity solutions that protect against evolving threats\n- Evidence: \"SecureNet's mission is to provide enterprise-grade cybersecurity solutions that protect against evolving threats\"\n- Action: [KEEP]\n- Item: world where cyber threats are eliminated\n- Evidence: \"The company's vision emphasizes a world where cyber threats are eliminated\"\n- Action: [DISCARD] (Reason: Vision, not mission)\n- Item: security-first thinking, customer protection, continuous innovation\n- Evidence: \"Core values include security-first thinking, customer protection, and continuous innovation\"\n- Action: [DISCARD] (Reason: Values, not mission)\n- Item: protect 1 million endpoints\n- Evidence: \"The company's target is to protect 1 million endpoints by 2025\"\n- Action: [DISCARD] (Reason: Target, not mission)\n- End of scan.\nFINAL ANSWER:\nSecureNet's mission is to provide enterprise-grade cybersecurity solutions that protect against evolving threats."
        )
    ]
    examples.extend(mission_examples)
    
    # 5. TEAM SIZE/HEADCOUNT (5 examples)
    team_size_examples = [
        create_example(
            "Knowledge context: TechCorp has a diverse workforce across multiple departments. The engineering team consists of 50 developers working on core products. The marketing team has 20 members responsible for brand and communications. The sales team includes 30 people focused on customer acquisition. The total company headcount is 100 employees.\n---\nQuestion: How many employees are in the engineering team?",
            "REASONING:\n- Item: 50 developers\n- Evidence: \"The engineering team consists of 50 developers working on core products\"\n- Action: [KEEP]\n- Item: 20 marketing members\n- Evidence: \"The marketing team has 20 members responsible for brand and communications\"\n- Action: [DISCARD] (Reason: Marketing team, not engineering team)\n- Item: 30 sales people\n- Evidence: \"The sales team includes 30 people focused on customer acquisition\"\n- Action: [DISCARD] (Reason: Sales team, not engineering team)\n- Item: 100 total employees\n- Evidence: \"The total company headcount is 100 employees\"\n- Action: [DISCARD] (Reason: Total headcount, not engineering team size)\n- End of scan.\nFINAL ANSWER:\nThe engineering team has 50 employees (developers)."
        ),
        create_example(
            "Knowledge context: DataFlow employs professionals across various functions. The product development team has 45 engineers and designers. The customer success team consists of 25 specialists. The operations team includes 15 staff members. The entire organization has 85 full-time employees.\n---\nQuestion: How many people are in the product development team?",
            "REASONING:\n- Item: 45 engineers and designers\n- Evidence: \"The product development team has 45 engineers and designers\"\n- Action: [KEEP]\n- Item: 25 customer success specialists\n- Evidence: \"The customer success team consists of 25 specialists\"\n- Action: [DISCARD] (Reason: Customer success team, not product development)\n- Item: 15 operations staff\n- Evidence: \"The operations team includes 15 staff members\"\n- Action: [DISCARD] (Reason: Operations team, not product development)\n- Item: 85 total employees\n- Evidence: \"The entire organization has 85 full-time employees\"\n- Action: [DISCARD] (Reason: Total employees, not product development team)\n- End of scan.\nFINAL ANSWER:\nThe product development team has 45 people (engineers and designers)."
        ),
        create_example(
            "Knowledge context: CloudScale has teams distributed across multiple locations. The research and development team consists of 60 scientists and engineers. The sales and marketing team has 35 professionals. The administrative and support staff includes 20 employees. The company's total workforce is 115 people.\n---\nQuestion: How many employees work in the research and development team?",
            "REASONING:\n- Item: 60 scientists and engineers\n- Evidence: \"The research and development team consists of 60 scientists and engineers\"\n- Action: [KEEP]\n- Item: 35 sales and marketing professionals\n- Evidence: \"The sales and marketing team has 35 professionals\"\n- Action: [DISCARD] (Reason: Sales and marketing team, not R&D)\n- Item: 20 administrative and support staff\n- Evidence: \"The administrative and support staff includes 20 employees\"\n- Action: [DISCARD] (Reason: Administrative staff, not R&D)\n- Item: 115 total workforce\n- Evidence: \"The company's total workforce is 115 people\"\n- Action: [DISCARD] (Reason: Total workforce, not R&D team)\n- End of scan.\nFINAL ANSWER:\nThe research and development team has 60 employees (scientists and engineers)."
        ),
        create_example(
            "Knowledge context: SecureNet maintains specialized teams for different functions. The cybersecurity team has 40 security experts and analysts. The customer support team consists of 18 representatives. The business development team includes 12 professionals. The total number of employees at SecureNet is 70.\n---\nQuestion: How many people are in the cybersecurity team?",
            "REASONING:\n- Item: 40 security experts and analysts\n- Evidence: \"The cybersecurity team has 40 security experts and analysts\"\n- Action: [KEEP]\n- Item: 18 customer support representatives\n- Evidence: \"The customer support team consists of 18 representatives\"\n- Action: [DISCARD] (Reason: Customer support team, not cybersecurity)\n- Item: 12 business development professionals\n- Evidence: \"The business development team includes 12 professionals\"\n- Action: [DISCARD] (Reason: Business development team, not cybersecurity)\n- Item: 70 total employees\n- Evidence: \"The total number of employees at SecureNet is 70\"\n- Action: [DISCARD] (Reason: Total employees, not cybersecurity team)\n- End of scan.\nFINAL ANSWER:\nThe cybersecurity team has 40 people (security experts and analysts)."
        ),
        create_example(
            "Knowledge context: AIForge has a growing team structure. The machine learning team consists of 55 data scientists and ML engineers. The platform engineering team has 30 software developers. The quality assurance team includes 10 testers. The company employs 95 people in total.\n---\nQuestion: How many employees are in the machine learning team?",
            "REASONING:\n- Item: 55 data scientists and ML engineers\n- Evidence: \"The machine learning team consists of 55 data scientists and ML engineers\"\n- Action: [KEEP]\n- Item: 30 platform engineering developers\n- Evidence: \"The platform engineering team has 30 software developers\"\n- Action: [DISCARD] (Reason: Platform engineering team, not machine learning)\n- Item: 10 QA testers\n- Evidence: \"The quality assurance team includes 10 testers\"\n- Action: [DISCARD] (Reason: QA team, not machine learning)\n- Item: 95 total employees\n- Evidence: \"The company employs 95 people in total\"\n- Action: [DISCARD] (Reason: Total employees, not machine learning team)\n- End of scan.\nFINAL ANSWER:\nThe machine learning team has 55 employees (data scientists and ML engineers)."
        )
    ]
    examples.extend(team_size_examples)
    
    # 6. SERVICES (vs PRODUCTS) (5 examples)
    services_examples = [
        create_example(
            "Knowledge context: CloudScale offers three main products: CloudScale Compute, CloudScale Storage, and CloudScale Analytics. The company also provides consulting services for cloud migration and implementation. CloudScale Support is available as a service for enterprise customers. Training and certification programs are offered as educational services.\n---\nQuestion: What services does CloudScale offer?",
            "REASONING:\n- Item: consulting services\n- Evidence: \"The company also provides consulting services for cloud migration and implementation\"\n- Action: [KEEP]\n- Item: CloudScale Support\n- Evidence: \"CloudScale Support is available as a service for enterprise customers\"\n- Action: [KEEP]\n- Item: training and certification programs\n- Evidence: \"Training and certification programs are offered as educational services\"\n- Action: [KEEP]\n- Item: CloudScale Compute\n- Evidence: \"CloudScale offers three main products: CloudScale Compute, CloudScale Storage, and CloudScale Analytics\"\n- Action: [DISCARD] (Reason: Product, not service)\n- Item: CloudScale Storage\n- Evidence: \"CloudScale offers three main products: CloudScale Compute, CloudScale Storage, and CloudScale Analytics\"\n- Action: [DISCARD] (Reason: Product, not service)\n- Item: CloudScale Analytics\n- Evidence: \"CloudScale offers three main products: CloudScale Compute, CloudScale Storage, and CloudScale Analytics\"\n- Action: [DISCARD] (Reason: Product, not service)\n- End of scan.\nFINAL ANSWER:\nCloudScale offers consulting services for cloud migration and implementation, CloudScale Support for enterprise customers, and training and certification programs as educational services."
        ),
        create_example(
            "Knowledge context: DataVault provides enterprise data security products including DataVault Encryption Suite and DataVault Backup System. The company offers managed security services for 24/7 monitoring and threat response. Professional services include security audits and compliance consulting. Implementation and integration services are available for new deployments.\n---\nQuestion: What services does DataVault provide?",
            "REASONING:\n- Item: managed security services\n- Evidence: \"The company offers managed security services for 24/7 monitoring and threat response\"\n- Action: [KEEP]\n- Item: professional services (security audits and compliance consulting)\n- Evidence: \"Professional services include security audits and compliance consulting\"\n- Action: [KEEP]\n- Item: implementation and integration services\n- Evidence: \"Implementation and integration services are available for new deployments\"\n- Action: [KEEP]\n- Item: DataVault Encryption Suite\n- Evidence: \"DataVault provides enterprise data security products including DataVault Encryption Suite and DataVault Backup System\"\n- Action: [DISCARD] (Reason: Product, not service)\n- Item: DataVault Backup System\n- Evidence: \"DataVault provides enterprise data security products including DataVault Encryption Suite and DataVault Backup System\"\n- Action: [DISCARD] (Reason: Product, not service)\n- End of scan.\nFINAL ANSWER:\nDataVault provides managed security services for 24/7 monitoring and threat response, professional services including security audits and compliance consulting, and implementation and integration services for new deployments."
        ),
        create_example(
            "Knowledge context: TechBuild offers software development products: TechBuild IDE, TechBuild Debugger, and TechBuild Profiler. The company provides cloud hosting services for application deployment. Maintenance and support services are available on subscription basis. Custom development services are offered for enterprise clients.\n---\nQuestion: What services does TechBuild offer?",
            "REASONING:\n- Item: cloud hosting services\n- Evidence: \"The company provides cloud hosting services for application deployment\"\n- Action: [KEEP]\n- Item: maintenance and support services\n- Evidence: \"Maintenance and support services are available on subscription basis\"\n- Action: [KEEP]\n- Item: custom development services\n- Evidence: \"Custom development services are offered for enterprise clients\"\n- Action: [KEEP]\n- Item: TechBuild IDE\n- Evidence: \"TechBuild offers software development products: TechBuild IDE, TechBuild Debugger, and TechBuild Profiler\"\n- Action: [DISCARD] (Reason: Product, not service)\n- Item: TechBuild Debugger\n- Evidence: \"TechBuild offers software development products: TechBuild IDE, TechBuild Debugger, and TechBuild Profiler\"\n- Action: [DISCARD] (Reason: Product, not service)\n- Item: TechBuild Profiler\n- Evidence: \"TechBuild offers software development products: TechBuild IDE, TechBuild Debugger, and TechBuild Profiler\"\n- Action: [DISCARD] (Reason: Product, not service)\n- End of scan.\nFINAL ANSWER:\nTechBuild offers cloud hosting services for application deployment, maintenance and support services on subscription basis, and custom development services for enterprise clients."
        ),
        create_example(
            "Knowledge context: AIForge markets AI development products: AIForge Studio, AIForge Runtime, and AIForge SDK. The company delivers AI consulting services to help organizations implement AI solutions. Training services are provided for teams to learn AI development. Managed AI infrastructure services are available for enterprise deployments.\n---\nQuestion: What services does AIForge deliver?",
            "REASONING:\n- Item: AI consulting services\n- Evidence: \"The company delivers AI consulting services to help organizations implement AI solutions\"\n- Action: [KEEP]\n- Item: training services\n- Evidence: \"Training services are provided for teams to learn AI development\"\n- Action: [KEEP]\n- Item: managed AI infrastructure services\n- Evidence: \"Managed AI infrastructure services are available for enterprise deployments\"\n- Action: [KEEP]\n- Item: AIForge Studio\n- Evidence: \"AIForge markets AI development products: AIForge Studio, AIForge Runtime, and AIForge SDK\"\n- Action: [DISCARD] (Reason: Product, not service)\n- Item: AIForge Runtime\n- Evidence: \"AIForge markets AI development products: AIForge Studio, AIForge Runtime, and AIForge SDK\"\n- Action: [DISCARD] (Reason: Product, not service)\n- Item: AIForge SDK\n- Evidence: \"AIForge markets AI development products: AIForge Studio, AIForge Runtime, and AIForge SDK\"\n- Action: [DISCARD] (Reason: Product, not service)\n- End of scan.\nFINAL ANSWER:\nAIForge delivers AI consulting services to help organizations implement AI solutions, training services for teams to learn AI development, and managed AI infrastructure services for enterprise deployments."
        ),
        create_example(
            "Knowledge context: SecureNet sells security products: SecureNet Firewall, SecureNet VPN, and SecureNet Antivirus. The company offers security monitoring services for continuous threat detection. Incident response services are available for security breaches. Security assessment services help organizations evaluate their security posture.\n---\nQuestion: What services does SecureNet offer?",
            "REASONING:\n- Item: security monitoring services\n- Evidence: \"The company offers security monitoring services for continuous threat detection\"\n- Action: [KEEP]\n- Item: incident response services\n- Evidence: \"Incident response services are available for security breaches\"\n- Action: [KEEP]\n- Item: security assessment services\n- Evidence: \"Security assessment services help organizations evaluate their security posture\"\n- Action: [KEEP]\n- Item: SecureNet Firewall\n- Evidence: \"SecureNet sells security products: SecureNet Firewall, SecureNet VPN, and SecureNet Antivirus\"\n- Action: [DISCARD] (Reason: Product, not service)\n- Item: SecureNet VPN\n- Evidence: \"SecureNet sells security products: SecureNet Firewall, SecureNet VPN, and SecureNet Antivirus\"\n- Action: [DISCARD] (Reason: Product, not service)\n- Item: SecureNet Antivirus\n- Evidence: \"SecureNet sells security products: SecureNet Firewall, SecureNet VPN, and SecureNet Antivirus\"\n- Action: [DISCARD] (Reason: Product, not service)\n- End of scan.\nFINAL ANSWER:\nSecureNet offers security monitoring services for continuous threat detection, incident response services for security breaches, and security assessment services to help organizations evaluate their security posture."
        )
    ]
    examples.extend(services_examples)
    
    # 7. DRAWBACKS/LIMITATIONS (5 examples)
    drawbacks_examples = [
        create_example(
            "Knowledge context: Cloud-based AI solutions present several notable drawbacks and limitations. Latency issues can occur when processing large datasets over network connections. Security risks exist when sensitive data is transmitted to external servers. Compliance challenges arise when data must remain within specific geographic boundaries. However, cloud solutions also offer benefits like scalability and cost efficiency.\n---\nQuestion: What are the drawbacks of cloud-based AI solutions?",
            "REASONING:\n- Item: latency issues\n- Evidence: \"Latency issues can occur when processing large datasets over network connections\"\n- Action: [KEEP]\n- Item: security risks\n- Evidence: \"Security risks exist when sensitive data is transmitted to external servers\"\n- Action: [KEEP]\n- Item: compliance challenges\n- Evidence: \"Compliance challenges arise when data must remain within specific geographic boundaries\"\n- Action: [KEEP]\n- Item: scalability and cost efficiency\n- Evidence: \"However, cloud solutions also offer benefits like scalability and cost efficiency\"\n- Action: [DISCARD] (Reason: Benefits, not drawbacks)\n- End of scan.\nFINAL ANSWER:\nThe drawbacks of cloud-based AI solutions include latency issues when processing large datasets over network connections, security risks when sensitive data is transmitted to external servers, and compliance challenges when data must remain within specific geographic boundaries."
        ),
        create_example(
            "Knowledge context: Legacy reporting systems have inherent limitations that impact business operations. Delayed decision-making results from slow data processing which negatively impacts market positioning. Reactive governance models emerge when organizations lack predictive information and respond only after challenges arise. Lack of predictive insights prevents businesses from benefiting from AI-driven forecasting. Modern systems offer advantages like real-time analytics and automated reporting.\n---\nQuestion: What are the limitations of legacy reporting systems?",
            "REASONING:\n- Item: delayed decision-making\n- Evidence: \"Delayed decision-making results from slow data processing which negatively impacts market positioning\"\n- Action: [KEEP]\n- Item: reactive governance models\n- Evidence: \"Reactive governance models emerge when organizations lack predictive information and respond only after challenges arise\"\n- Action: [KEEP]\n- Item: lack of predictive insights\n- Evidence: \"Lack of predictive insights prevents businesses from benefiting from AI-driven forecasting\"\n- Action: [KEEP]\n- Item: real-time analytics and automated reporting\n- Evidence: \"Modern systems offer advantages like real-time analytics and automated reporting\"\n- Action: [DISCARD] (Reason: Advantages of modern systems, not limitations of legacy systems)\n- End of scan.\nFINAL ANSWER:\nThe limitations of legacy reporting systems include delayed decision-making from slow data processing, reactive governance models when organizations lack predictive information, and lack of predictive insights that prevent businesses from benefiting from AI-driven forecasting."
        ),
        create_example(
            "Knowledge context: Manual data reconciliation processes face significant challenges. Time-consuming operations require extensive human effort and are prone to errors. Limited scalability prevents organizations from handling large volumes of data efficiently. High operational costs result from the need for dedicated staff. Automated systems provide benefits such as speed and accuracy improvements.\n---\nQuestion: What are the drawbacks of manual data reconciliation processes?",
            "REASONING:\n- Item: time-consuming operations\n- Evidence: \"Time-consuming operations require extensive human effort and are prone to errors\"\n- Action: [KEEP]\n- Item: limited scalability\n- Evidence: \"Limited scalability prevents organizations from handling large volumes of data efficiently\"\n- Action: [KEEP]\n- Item: high operational costs\n- Evidence: \"High operational costs result from the need for dedicated staff\"\n- Action: [KEEP]\n- Item: speed and accuracy improvements\n- Evidence: \"Automated systems provide benefits such as speed and accuracy improvements\"\n- Action: [DISCARD] (Reason: Benefits of automated systems, not drawbacks of manual processes)\n- End of scan.\nFINAL ANSWER:\nThe drawbacks of manual data reconciliation processes include time-consuming operations that require extensive human effort and are prone to errors, limited scalability that prevents handling large volumes efficiently, and high operational costs from the need for dedicated staff."
        ),
        create_example(
            "Knowledge context: Static spreadsheet-based reporting has several limitations. Data silos prevent comprehensive analysis across departments. Version control issues arise when multiple users edit files simultaneously. Limited collaboration capabilities hinder real-time team coordination. Modern cloud-based platforms offer advantages including real-time collaboration and centralized data management.\n---\nQuestion: What are the limitations of static spreadsheet-based reporting?",
            "REASONING:\n- Item: data silos\n- Evidence: \"Data silos prevent comprehensive analysis across departments\"\n- Action: [KEEP]\n- Item: version control issues\n- Evidence: \"Version control issues arise when multiple users edit files simultaneously\"\n- Action: [KEEP]\n- Item: limited collaboration capabilities\n- Evidence: \"Limited collaboration capabilities hinder real-time team coordination\"\n- Action: [KEEP]\n- Item: real-time collaboration and centralized data management\n- Evidence: \"Modern cloud-based platforms offer advantages including real-time collaboration and centralized data management\"\n- Action: [DISCARD] (Reason: Advantages of modern platforms, not limitations of spreadsheets)\n- End of scan.\nFINAL ANSWER:\nThe limitations of static spreadsheet-based reporting include data silos that prevent comprehensive analysis across departments, version control issues when multiple users edit files simultaneously, and limited collaboration capabilities that hinder real-time team coordination."
        ),
        create_example(
            "Knowledge context: On-premises infrastructure solutions have certain drawbacks that organizations must consider. High upfront capital investment is required for hardware and software purchases. Maintenance responsibilities fall entirely on the organization's IT team. Limited flexibility makes it difficult to scale resources up or down quickly. Cloud alternatives provide benefits such as pay-as-you-go pricing and automatic scaling.\n---\nQuestion: What are the drawbacks of on-premises infrastructure solutions?",
            "REASONING:\n- Item: high upfront capital investment\n- Evidence: \"High upfront capital investment is required for hardware and software purchases\"\n- Action: [KEEP]\n- Item: maintenance responsibilities\n- Evidence: \"Maintenance responsibilities fall entirely on the organization's IT team\"\n- Action: [KEEP]\n- Item: limited flexibility\n- Evidence: \"Limited flexibility makes it difficult to scale resources up or down quickly\"\n- Action: [KEEP]\n- Item: pay-as-you-go pricing and automatic scaling\n- Evidence: \"Cloud alternatives provide benefits such as pay-as-you-go pricing and automatic scaling\"\n- Action: [DISCARD] (Reason: Benefits of cloud alternatives, not drawbacks of on-premises)\n- End of scan.\nFINAL ANSWER:\nThe drawbacks of on-premises infrastructure solutions include high upfront capital investment required for hardware and software purchases, maintenance responsibilities that fall entirely on the organization's IT team, and limited flexibility that makes it difficult to scale resources up or down quickly."
        )
    ]
    examples.extend(drawbacks_examples)
    
    # 8. OFFICE LOCATIONS (4 examples)
    locations_examples = [
        create_example(
            "Knowledge context: TechCorp has a global presence with offices in multiple cities. The company maintains offices in San Francisco, New York, London, and Tokyo. The San Francisco location serves as the corporate headquarters. The New York office opened in 2019 to serve East Coast clients. The London office handles European operations. The Tokyo office was established in 2021 for Asia-Pacific markets.\n---\nQuestion: Where are TechCorp's offices located?",
            "REASONING:\n- Item: San Francisco\n- Evidence: \"The company maintains offices in San Francisco, New York, London, and Tokyo\"\n- Action: [KEEP]\n- Item: New York\n- Evidence: \"The company maintains offices in San Francisco, New York, London, and Tokyo\"\n- Action: [KEEP]\n- Item: London\n- Evidence: \"The company maintains offices in San Francisco, New York, London, and Tokyo\"\n- Action: [KEEP]\n- Item: Tokyo\n- Evidence: \"The company maintains offices in San Francisco, New York, London, and Tokyo\"\n- Action: [KEEP]\n- Item: 2019\n- Evidence: \"The New York office opened in 2019\"\n- Action: [DISCARD] (Reason: Date, not location)\n- Item: 2021\n- Evidence: \"The Tokyo office was established in 2021\"\n- Action: [DISCARD] (Reason: Date, not location)\n- End of scan.\nFINAL ANSWER:\nTechCorp's offices are located in San Francisco, New York, London, and Tokyo."
        ),
        create_example(
            "Knowledge context: DataFlow operates from multiple regional offices worldwide. The company has offices in Seattle, Boston, Berlin, and Singapore. The Seattle office is the main headquarters. The Boston location focuses on research and development. The Berlin office serves European customers. The Singapore office manages operations in Southeast Asia.\n---\nQuestion: Where does DataFlow have offices?",
            "REASONING:\n- Item: Seattle\n- Evidence: \"The company has offices in Seattle, Boston, Berlin, and Singapore\"\n- Action: [KEEP]\n- Item: Boston\n- Evidence: \"The company has offices in Seattle, Boston, Berlin, and Singapore\"\n- Action: [KEEP]\n- Item: Berlin\n- Evidence: \"The company has offices in Seattle, Boston, Berlin, and Singapore\"\n- Action: [KEEP]\n- Item: Singapore\n- Evidence: \"The company has offices in Seattle, Boston, Berlin, and Singapore\"\n- Action: [KEEP]\n- End of scan.\nFINAL ANSWER:\nDataFlow has offices in Seattle, Boston, Berlin, and Singapore."
        ),
        create_example(
            "Knowledge context: CloudScale maintains a distributed office network across continents. The company operates offices in Austin, Chicago, Paris, and Sydney. The Austin location serves as the primary headquarters. The Chicago office handles Midwest operations. The Paris office manages European business. The Sydney office covers Australia and New Zealand markets.\n---\nQuestion: What cities have CloudScale offices?",
            "REASONING:\n- Item: Austin\n- Evidence: \"The company operates offices in Austin, Chicago, Paris, and Sydney\"\n- Action: [KEEP]\n- Item: Chicago\n- Evidence: \"The company operates offices in Austin, Chicago, Paris, and Sydney\"\n- Action: [KEEP]\n- Item: Paris\n- Evidence: \"The company operates offices in Austin, Chicago, Paris, and Sydney\"\n- Action: [KEEP]\n- Item: Sydney\n- Evidence: \"The company operates offices in Austin, Chicago, Paris, and Sydney\"\n- Action: [KEEP]\n- End of scan.\nFINAL ANSWER:\nCloudScale has offices in Austin, Chicago, Paris, and Sydney."
        ),
        create_example(
            "Knowledge context: SecureNet has established offices in key technology hubs. The company maintains locations in Denver, Atlanta, Amsterdam, and Hong Kong. The Denver office is the corporate headquarters. The Atlanta location focuses on sales and customer success. The Amsterdam office serves European markets. The Hong Kong office handles Asia-Pacific operations.\n---\nQuestion: Where are SecureNet's offices located?",
            "REASONING:\n- Item: Denver\n- Evidence: \"The company maintains locations in Denver, Atlanta, Amsterdam, and Hong Kong\"\n- Action: [KEEP]\n- Item: Atlanta\n- Evidence: \"The company maintains locations in Denver, Atlanta, Amsterdam, and Hong Kong\"\n- Action: [KEEP]\n- Item: Amsterdam\n- Evidence: \"The company maintains locations in Denver, Atlanta, Amsterdam, and Hong Kong\"\n- Action: [KEEP]\n- Item: Hong Kong\n- Evidence: \"The company maintains locations in Denver, Atlanta, Amsterdam, and Hong Kong\"\n- Action: [KEEP]\n- End of scan.\nFINAL ANSWER:\nSecureNet's offices are located in Denver, Atlanta, Amsterdam, and Hong Kong."
        )
    ]
    examples.extend(locations_examples)
    
    # 9. SPECIFIC ROLES (CTO, COO, etc.) (4 examples)
    roles_examples = [
        create_example(
            "Knowledge context: TechVentures has a strong leadership team. Sarah Johnson is the Chief Technology Officer at TechVentures, leading all engineering and technical initiatives. Mark Williams serves as the Head of Engineering, reporting to the CTO. Robert Kim is the Chief Financial Officer, managing financial operations. The company's technical strategy is driven by the CTO.\n---\nQuestion: Who is the CTO of TechVentures?",
            "REASONING:\n- Item: Sarah Johnson\n- Evidence: \"Sarah Johnson is the Chief Technology Officer at TechVentures\"\n- Action: [KEEP]\n- Item: Mark Williams\n- Evidence: \"Mark Williams serves as the Head of Engineering, reporting to the CTO\"\n- Action: [DISCARD] (Reason: Head of Engineering, not CTO)\n- Item: Robert Kim\n- Evidence: \"Robert Kim is the Chief Financial Officer, managing financial operations\"\n- Action: [DISCARD] (Reason: CFO, not CTO)\n- End of scan.\nFINAL ANSWER:\nThe CTO of TechVentures is Sarah Johnson."
        ),
        create_example(
            "Knowledge context: DataSystems employs experienced executives across key functions. Michael Chen is the Chief Operating Officer of DataSystems, responsible for day-to-day operations and strategic execution. Lisa Park serves as the Vice President of Operations, supporting the COO. David Martinez is the Chief Executive Officer, providing overall strategic direction. Operational excellence is a priority for the COO.\n---\nQuestion: Who is the COO of DataSystems?",
            "REASONING:\n- Item: Michael Chen\n- Evidence: \"Michael Chen is the Chief Operating Officer of DataSystems\"\n- Action: [KEEP]\n- Item: Lisa Park\n- Evidence: \"Lisa Park serves as the Vice President of Operations, supporting the COO\"\n- Action: [DISCARD] (Reason: VP of Operations, not COO)\n- Item: David Martinez\n- Evidence: \"David Martinez is the Chief Executive Officer, providing overall strategic direction\"\n- Action: [DISCARD] (Reason: CEO, not COO)\n- End of scan.\nFINAL ANSWER:\nThe COO of DataSystems is Michael Chen."
        ),
        create_example(
            "Knowledge context: CloudTech has a diverse executive leadership team. Jennifer Wong is the Chief Marketing Officer at CloudTech, leading brand strategy and market expansion. Thomas Lee serves as the Director of Marketing, working under the CMO. Amanda Rodriguez is the Chief Revenue Officer, focusing on sales and revenue growth. Marketing initiatives are overseen by the CMO.\n---\nQuestion: Who is the CMO of CloudTech?",
            "REASONING:\n- Item: Jennifer Wong\n- Evidence: \"Jennifer Wong is the Chief Marketing Officer at CloudTech\"\n- Action: [KEEP]\n- Item: Thomas Lee\n- Evidence: \"Thomas Lee serves as the Director of Marketing, working under the CMO\"\n- Action: [DISCARD] (Reason: Director of Marketing, not CMO)\n- Item: Amanda Rodriguez\n- Evidence: \"Amanda Rodriguez is the Chief Revenue Officer, focusing on sales and revenue growth\"\n- Action: [DISCARD] (Reason: CRO, not CMO)\n- End of scan.\nFINAL ANSWER:\nThe CMO of CloudTech is Jennifer Wong."
        ),
        create_example(
            "Knowledge context: SecureNet's leadership includes specialized executives. Kevin Park is the Chief Security Officer of SecureNet, responsible for security strategy and threat management. Maria Garcia serves as the Security Director, reporting to the CSO. James Wilson is the Chief Technology Officer, leading technical development. Security operations are managed by the CSO.\n---\nQuestion: Who is the CSO of SecureNet?",
            "REASONING:\n- Item: Kevin Park\n- Evidence: \"Kevin Park is the Chief Security Officer of SecureNet\"\n- Action: [KEEP]\n- Item: Maria Garcia\n- Evidence: \"Maria Garcia serves as the Security Director, reporting to the CSO\"\n- Action: [DISCARD] (Reason: Security Director, not CSO)\n- Item: James Wilson\n- Evidence: \"James Wilson is the Chief Technology Officer, leading technical development\"\n- Action: [DISCARD] (Reason: CTO, not CSO)\n- End of scan.\nFINAL ANSWER:\nThe CSO of SecureNet is Kevin Park."
        )
    ]
    examples.extend(roles_examples)
    
    # Verify we have exactly 43 examples
    if len(examples) != 43:
        print(f"⚠️  Warning: Created {len(examples)} examples, expected 43")
    
    return examples

def main():
    """Main function to diversify the dataset"""
    print("=" * 80)
    print("QUERY DIVERSIFICATION SCRIPT")
    print("=" * 80)
    print()
    
    # Load dataset
    print("Loading dataset...")
    with open('rag_cot_training_dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✅ Loaded {len(dataset)} examples")
    
    # Find co-founder indices to replace
    cofounder_indices = []
    for i, example in enumerate(dataset):
        query = example['messages'][1]['content'].lower()
        if 'co-founder' in query or 'cofounder' in query:
            cofounder_indices.append(i)
    
    # Keep first 6 (indices 0-5) as real-world examples
    indices_to_replace = [idx for idx in cofounder_indices if idx >= 6][:43]
    
    print(f"📊 Found {len(cofounder_indices)} co-founder queries")
    print(f"   Keeping indices 0-5 (real-world examples)")
    print(f"   Will replace {len(indices_to_replace)} queries")
    print()
    
    # Create diverse examples
    print("Creating diverse examples...")
    diverse_examples = create_diverse_examples()
    
    if len(diverse_examples) < 43:
        print(f"⚠️  Only {len(diverse_examples)} examples created, need 43")
        print("   Please complete the create_diverse_examples() function")
        return
    
    # Replace examples
    print("Replacing examples in dataset...")
    for i, idx in enumerate(indices_to_replace):
        dataset[idx] = diverse_examples[i]
        if (i + 1) % 10 == 0:
            print(f"   Replaced {i + 1}/{len(indices_to_replace)} examples...")
    
    # Save updated dataset
    print("Saving updated dataset...")
    with open('rag_cot_training_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print("✅ Dataset diversified successfully!")
    print()
    print("New query distribution:")
    # Count query types
    query_types = {}
    for example in dataset:
        query = example['messages'][1]['content'].lower()
        if 'co-founder' in query or 'cofounder' in query:
            query_types['co-founder'] = query_types.get('co-founder', 0) + 1
        elif 'programming' in query or 'language' in query or 'technology' in query:
            query_types['technologies'] = query_types.get('technologies', 0) + 1
        elif 'revenue' in query or 'financial' in query:
            query_types['revenue'] = query_types.get('revenue', 0) + 1
        # Add more type checks...
    
    for qtype, count in sorted(query_types.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(dataset)) * 100
        print(f"   - {qtype}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
