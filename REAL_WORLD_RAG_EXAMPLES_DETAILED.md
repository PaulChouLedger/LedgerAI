# 6 New Real-World RAG Examples - Complete Detailed Breakdown

## Summary Statistics

- **Total Examples**: 6
- **Total Items Processed**: 20
- **✅ [KEEP] Items**: 16
- **❌ [DISCARD] Items**: 4
- **Total Context Chunks**: 12
- **Average Chunks per Example**: 2.0
- **Average Items per Example**: 3.3
- **[KEEP] Rate**: 80.0%

---

## Example 1: Who are the co-founders of LedgerAI?

**Type**: Multi-chunk co-founders extraction with headers/metadata  
**Context Chunks**: 3  
**Items Processed**: 8 (4 KEEP, 4 DISCARD)

### Query
```
Who are the co-founders of LedgerAI?
```

### Context Chunks (3 total)
- **Chunk 1** (3,594 chars): Contains Bob Carella, David Lara, Jorge Guinovart, Will Specht information
- **Chunk 2** (2,065 chars): Contains Paul Chou and Bob Carella information
- **Chunk 3** (3,099 chars): Contains Albert Soler, Peter Moeller, Liam Hugill information

### Complete Reasoning

**REASONING:**
- **Item**: Bob Carella
  - **Evidence**: "As Co-Founder and Chief Financial Officer of LedgerAI"
  - **Action**: [KEEP]
  - **Reason**: Evidence contains 'Co-Founder', so this person IS a co-founder.

- **Item**: David Lara
  - **Evidence**: "As Co-Founder and Chief Operating Officer of LedgerAI"
  - **Action**: [KEEP]
  - **Reason**: Evidence contains 'Co-Founder', so this person IS a co-founder. Ignore header 'AURA VISION...'.

- **Item**: Jorge Guinovart
  - **Evidence**: "As Co-Founder and Chief Marketing Officer of LedgerAI"
  - **Action**: [KEEP]
  - **Reason**: Evidence contains 'Co-Founder', so this person IS a co-founder.

- **Item**: Will Specht
  - **Evidence**: "leading LedgerAI's cutting-edge engineering efforts as Head of Engineering"
  - **Action**: [DISCARD]
  - **Reason**: Evidence does not mention 'Co-Founder', only Head of Engineering role.

- **Item**: Paul Chou
  - **Evidence**: "As CEO and Co-Founder of LedgerAI"
  - **Action**: [KEEP]
  - **Reason**: Evidence contains 'Co-Founder', so this person IS a co-founder.

- **Item**: Albert Soler
  - **Evidence**: "bringing unparalleled expertise in litigation, intellectual property, and business law to LedgerAI as External Counsel & Advisor"
  - **Action**: [DISCARD]
  - **Reason**: Evidence does not mention 'Co-Founder', only External Counsel & Advisor role.

- **Item**: Peter Moeller
  - **Evidence**: "serving as Business Development Lead at LedgerAI"
  - **Action**: [DISCARD]
  - **Reason**: Evidence does not mention 'Co-Founder', only Business Development Lead role.

- **Item**: Liam Hugill
  - **Evidence**: "being a natural fit as LedgerAI's Ambassador of Influence and Engagement"
  - **Action**: [DISCARD]
  - **Reason**: Evidence does not mention 'Co-Founder', only Ambassador role.

- **End of scan.**

### Final Answer
```
The co-founders of LedgerAI are Paul Chou, Bob Carella, David Lara, and Jorge Guinovart.
```

### Verification
- ✅ **4 [KEEP] items** in REASONING
- ✅ **All 4 [KEEP] items** included in FINAL ANSWER
- ✅ **No [DISCARD] items** in FINAL ANSWER (correct!)
- ✅ **Evidence extraction**: Clean quotes, headers filtered out

**Key Training Points:**
- Multi-chunk scanning across 3 chunks
- Header/metadata filtering ("AURA VISION..." ignored in evidence)
- Co-founder identification rule: "Co-Founder" in evidence → [KEEP]
- Proper DISCARD of non-co-founders (Will Specht, Albert Soler, Peter Moeller, Liam Hugill)
- All [KEEP] items included in FINAL ANSWER

---

## Example 2: Do you know who David Lara is?

**Type**: Individual person query  
**Context Chunks**: 1  
**Items Processed**: 1 (1 KEEP, 0 DISCARD)

### Query
```
Do you know who David Lara is?
```

### Context Chunks (1 total)
- **Chunk 1** (3,594 chars): Contains full David Lara description with role, background, experience, education

### Complete Reasoning

**REASONING:**
- **Item**: David Lara
  - **Evidence**: "David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management, bridging the gap between technology, operations, and policymaking. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making."
  - **Action**: [KEEP]
  - **Reason**: Information about David Lara, his role at LedgerAI, and his background.

- **End of scan.**

### Final Answer
```
David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making. He is also the CEO of Petra Capital & Advisory, focusing on AI technology and fintech investments, and Co-Founder of SuperCity AI. He has extensive experience in both public and private sectors, having served as a Partner at Ichor Strategies (2020–2023) and held senior leadership roles in New York's city and state governments. David holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas.
```

### Verification
- ✅ **1 [KEEP] item** in REASONING
- ✅ **All 1 [KEEP] items** included in FINAL ANSWER
- ✅ **Comprehensive answer** includes role, background, experience, education

**Key Training Points:**
- Single-chunk person information extraction
- Comprehensive evidence extraction (role, background, experience)
- Complete FINAL ANSWER with all relevant information

---

## Example 3: Who's the CFO of Ledger AI?

**Type**: Role-specific query requiring multi-chunk scanning  
**Context Chunks**: 3  
**Items Processed**: 2 (2 KEEP, 0 DISCARD)

### Query
```
Who's the CFO of Ledger AI?
```

### Context Chunks (3 total)
- **Chunk 1** (3,292 chars): Disclaimer and legal text (no CFO information)
- **Chunk 2** (3,594 chars): Contains Bob Carella information with financial operations
- **Chunk 3** (3,099 chars): Contains Albert Soler, Peter Moeller, Liam Hugill (no CFO information)

### Complete Reasoning

**REASONING:**
- **Item**: Bob Carella
  - **Evidence**: "Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders."
  - **Action**: [KEEP]
  - **Reason**: Mention of Bob and financial operations, but need to find explicit CFO title. Scanning further...

- **Item**: Bob Carella
  - **Evidence**: "As Co-Founder and Chief Financial Officer of LedgerAI"
  - **Action**: [KEEP]
  - **Reason**: Explicit mention of "Chief Financial Officer of LedgerAI".

- **End of scan.**

### Final Answer
```
The CFO of LedgerAI is Bob Carella. He is Co-Founder and Chief Financial Officer of LedgerAI, where he architects the company's financial strategy, tokenomics, and investment framework. Previously, as Global Head of Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs.
```

### Verification
- ✅ **2 [KEEP] items** in REASONING (same person found twice - partial match then complete match)
- ✅ **All [KEEP] items** included in FINAL ANSWER
- ✅ **Progressive refinement**: Found partial match, then scanned further for explicit title

**Key Training Points:**
- Multi-chunk scanning to find specific role
- Progressive refinement pattern: find partial match, then scan for complete match
- Evidence found across chunks (Bob's financial operations in one chunk, CFO title in another)
- Complete FINAL ANSWER with role and background

---

## Example 4: Where did David Lara go to school?

**Type**: Educational background extraction  
**Context Chunks**: 1  
**Items Processed**: 1 (1 KEEP, 0 DISCARD)

### Query
```
Where did David Lara go to school?
```

### Context Chunks (1 total)
- **Chunk 1** (3,594 chars): Contains David Lara description including educational background

### Complete Reasoning

**REASONING:**
- **Item**: David Lara education
  - **Evidence**: "David holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas"
  - **Action**: [KEEP]
  - **Reason**: Explicit mention of David Lara's educational background.

- **End of scan.**

### Final Answer
```
David Lara holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas.
```

### Verification
- ✅ **1 [KEEP] item** in REASONING
- ✅ **All [KEEP] items** included in FINAL ANSWER
- ✅ **Specific information extraction**: Education details extracted from longer description

**Key Training Points:**
- Specific information extraction (education from person description)
- Verbatim evidence extraction with exact degree names and universities
- Concise FINAL ANSWER with only relevant information

---

## Example 5: What do you know about the ledger token?

**Type**: Multi-item token information extraction  
**Context Chunks**: 3  
**Items Processed**: 4 (4 KEEP, 0 DISCARD)

### Query
```
What do you know about the ledger token?
```

### Context Chunks (3 total)
- **Chunk 1** (3,399 chars): Contains token definition, usage, and economy information
- **Chunk 2** (3,643 chars): Contains additional token information and capabilities
- **Chunk 3** (367 chars): Contains Liam Hugill information (no token information)

### Complete Reasoning

**REASONING:**
- **Item**: $LEDGER token information
  - **Evidence**: "$LEDGER, an ERC-20 token designed for frictionless access to computing power, AI-driven insights, and hardware integration"
  - **Action**: [KEEP]
  - **Reason**: Information about the ledger token.

- **Item**: Token usage
  - **Evidence**: "Businesses purchase $LEDGER tokens either directly from LedgerAI or on the open market to access compute power, AI services, and infrastructure"
  - **Action**: [KEEP]
  - **Reason**: How the token is used.

- **Item**: Token economy
  - **Evidence**: "Clients use the $LEDGER tokens to access AuraVision services and features fueling the ecosystem, while mechanisms such as buybacks and revenue sharing drive long-term sustainability and growth"
  - **Action**: [KEEP]
  - **Reason**: Token economy model.

- **Item**: The Ledger ERC-20 Token
  - **Evidence**: "The Ledger ERC-20 Token ($LEDGER) extends these capabilities by delivering tokenized access to AI-driven insights, enabling seamless integration across a variety of industries and establishing a viable and scalable ecosystem for enterprise intelligence"
  - **Action**: [KEEP]
  - **Reason**: Additional information about the token.

- **End of scan.**

### Final Answer
```
The Ledger token ($LEDGER) is an ERC-20 token designed for frictionless access to computing power, AI-driven insights, and hardware integration. It powers LedgerAI's tokenized revenue model, allowing businesses to purchase $LEDGER tokens either directly from LedgerAI or on the open market to access compute power, AI services, and infrastructure. Clients use $LEDGER tokens to access AuraVision services, and the token system includes mechanisms such as buybacks and revenue sharing to drive long-term sustainability and growth. The $LEDGER token extends LedgerAI's capabilities by delivering tokenized access to AI-driven insights, enabling seamless integration across industries and establishing a viable and scalable ecosystem for enterprise intelligence.
```

### Verification
- ✅ **4 [KEEP] items** in REASONING
- ✅ **All 4 [KEEP] items** included in FINAL ANSWER
- ✅ **Multi-item extraction**: Token definition, usage, economy, and capabilities

**Key Training Points:**
- Multi-item information extraction from multiple chunks
- Different aspects of same topic extracted separately (definition, usage, economy, capabilities)
- Complete FINAL ANSWER combining all aspects
- Multi-chunk scanning (token information found in chunks 1 and 2)

---

## Example 6: What are the benefits of localized?

**Type**: Benefits/features extraction from single chunk  
**Context Chunks**: 1  
**Items Processed**: 4 (4 KEEP, 0 DISCARD)

### Query
```
What are the benefits of localized?
```

### Context Chunks (1 total)
- **Chunk 1** (3,642 chars): Contains benefits of local AI processing, decentralized infrastructure, data recovery, and cloud comparison

### Complete Reasoning

**REASONING:**
- **Item**: Local AI benefits
  - **Evidence**: "On-Premises AI Processing – AuraVision operates within the Aura Network, running locally on secure business hardware. This ensures that sensitive data never leaves the organization's premises, eliminating reliance on centralized data processors"
  - **Action**: [KEEP]
  - **Reason**: Benefits of localized/local AI processing - data privacy and security.

- **Item**: Decentralized AI benefits
  - **Evidence**: "Decentralized AI Infrastructure – AuraVision leverages blockchain encryption to enhance security, automate compliance, and protect against data loss"
  - **Action**: [KEEP]
  - **Reason**: Benefits of decentralized/localized AI - security and compliance.

- **Item**: Data recovery benefits
  - **Evidence**: "Self-Destruct & Recovery Mechanism – If an AuraVision hardware device is lost, stolen, or compromised, it can self-destruct, rendering it useless to unauthorized parties. Importantly, businesses can securely recover data through Hammerhead's blockchain encryption, ensuring continuity without exposing sensitive information"
  - **Action**: [KEEP]
  - **Reason**: Benefits of localized system - data recovery and security.

- **Item**: Local AI advantages over cloud
  - **Evidence**: "Relying on cloud-based AI solutions presents several notable and important drawbacks and limitations, including latency issues, security risks, and compliance challenges"
  - **Action**: [KEEP]
  - **Reason**: Implies benefits of local AI - avoiding cloud drawbacks.

- **End of scan.**

### Final Answer
```
The benefits of localized (local AI) processing include: On-Premises AI Processing – AuraVision operates locally on secure business hardware, ensuring that sensitive data never leaves the organization's premises and eliminating reliance on centralized data processors. Decentralized AI Infrastructure – Leverages blockchain encryption to enhance security, automate compliance, and protect against data loss. Self-Destruct & Recovery Mechanism – If a device is lost, stolen, or compromised, it can self-destruct while allowing businesses to securely recover data through blockchain encryption, ensuring continuity without exposing sensitive information. Local AI also avoids the drawbacks of cloud-based solutions, including latency issues, security risks, and compliance challenges.
```

### Verification
- ✅ **4 [KEEP] items** in REASONING
- ✅ **All 4 [KEEP] items** included in FINAL ANSWER
- ✅ **Multi-benefit extraction**: 4 distinct benefits extracted and included

**Key Training Points:**
- Multi-benefit extraction from single chunk
- Benefits organized and presented clearly in FINAL ANSWER
- Implied benefit extraction (local AI advantages inferred from cloud drawbacks)
- Complete FINAL ANSWER with all identified benefits

---

## Overall Training Value

### Key Features Demonstrated

1. **Real-World RAG Chunks**
   - Headers/metadata (AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 23/24/25)
   - Page numbers and legal disclaimers
   - Fragmented contexts across multiple chunks

2. **Multi-Chunk Scanning**
   - Example 1: 3 chunks scanned for co-founders
   - Example 3: 3 chunks scanned for CFO (progressive refinement)
   - Example 5: 3 chunks scanned for token information

3. **Co-Founder Identification Rules**
   - "Co-Founder" in evidence → [KEEP]
   - No "Co-Founder" in evidence → [DISCARD]
   - Header filtering (ignore "AURA VISION..." in evidence)

4. **Role-Specific Extraction**
   - CFO extraction with multi-chunk scanning
   - Progressive refinement pattern (partial match → complete match)

5. **Educational Background Extraction**
   - Specific information extraction from longer descriptions
   - Verbatim evidence with exact degree names and universities

6. **Multi-Item Information Extraction**
   - Token example: 4 different aspects (definition, usage, economy, capabilities)
   - Benefits example: 4 distinct benefits extracted

7. **Complete REASONING**
   - Verbatim evidence extraction
   - Clear KEEP/DISCARD decisions with reasoning
   - Proper scanning across all chunks

8. **Complete FINAL ANSWER**
   - All [KEEP] items included
   - No [DISCARD] items in FINAL ANSWER
   - Comprehensive answers with relevant context

### Statistics

- **Total items processed**: 20
- **KEEP rate**: 80.0% (16/20)
- **DISCARD rate**: 20.0% (4/20)
- **Average chunks per example**: 2.0
- **Average items per example**: 3.3
- **Examples with multi-chunk contexts**: 3 (Examples 1, 3, 5)
- **Examples with single-chunk contexts**: 3 (Examples 2, 4, 6)
- **Examples with DISCARD items**: 1 (Example 1)
- **Examples with only KEEP items**: 5 (Examples 2, 3, 4, 5, 6)

### Quality Checks

- ✅ All examples include enhanced system prompt
- ✅ All examples show complete REASONING with evidence
- ✅ All examples show complete FINAL ANSWER
- ✅ All [KEEP] items verified in FINAL ANSWER
- ✅ No [DISCARD] items in FINAL ANSWER
- ✅ Evidence extraction excludes headers/metadata where appropriate
- ✅ Multi-chunk scanning demonstrated in complex queries
- ✅ Progressive refinement pattern shown (Example 3)

---

## Ready for Training

This dataset of 6 real-world examples, combined with the existing 165 enhanced examples, provides comprehensive training coverage for:

1. Real-world RAG scenarios with headers/metadata
2. Multi-chunk contexts requiring complete scanning
3. Co-founder identification with explicit rules
4. Role-specific extraction with progressive refinement
5. Educational background extraction
6. Multi-item information extraction
7. Benefits/features extraction
8. Proper DISCARD enforcement
9. Complete FINAL ANSWER generation

Total dataset size: **171 examples** (165 enhanced + 6 new real-world)
