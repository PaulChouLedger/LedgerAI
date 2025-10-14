#!/usr/bin/env python3
"""
Comprehensive Medical Data Ingestion System for RAG

Features:
1. Automated scraping of medical guidelines from authoritative sources
2. Medical document chunking optimized for clinical context
3. Automated updates for staying current with medical literature
4. Offline storage and indexing for clinician mode
5. Integration with existing RAG system

Supported Sources:
- PubMed/MEDLINE abstracts
- Clinical practice guidelines (ACP, AAFP, etc.)
- FDA drug information
- CDC guidelines
- WHO guidelines
- Medical journal articles
- UpToDate-style content (when accessible)

Usage:
python3 medical_data_ingestion.py --scrape --update --rebuild
"""

import os
import sys
import json
import requests
import time
import re
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import PyPDF2
import docx
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch

# Import existing RAG components
sys.path.append('rag-container')
try:
    from rag import AuraRAG
except ImportError:
    print("Warning: Could not import RAG components, some features may be limited")

class MedicalDataIngester:
    """
    Comprehensive medical data ingestion system for clinician RAG mode
    """

    def __init__(self, data_root="data"):
        self.data_root = Path(data_root)
        self.medical_dir = self.data_root / "medical"
        self.sources_dir = self.medical_dir / "sources"
        self.guidelines_dir = self.medical_dir / "guidelines"
        self.journals_dir = self.medical_dir / "journals"
        self.drugs_dir = self.medical_dir / "drugs"
        self.state_file = self.medical_dir / "ingestion_state.json"

        # Create directories
        for dir_path in [self.medical_dir, self.sources_dir, self.guidelines_dir,
                        self.journals_dir, self.drugs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Medical-specific chunking parameters
        self.chunk_size = 800  # Smaller chunks for medical precision
        self.chunk_overlap = 100  # More overlap for context preservation

        # Load state
        self.state = self.load_state()

        # Initialize medical sources
        self.pubmed_api_key = os.getenv('PUBMED_API_KEY')
        self.uptodate_credentials = os.getenv('UPTODATE_CREDENTIALS')  # If available

        print("🏥 Medical Data Ingester initialized")

    def load_state(self) -> Dict:
        """Load ingestion state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "last_update": None,
            "sources_processed": {},
            "total_articles": 0,
            "total_guidelines": 0
        }

    def save_state(self):
        """Save ingestion state"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            print(f"❌ Error saving state: {e}")

    def scrape_pubmed_articles(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Scrape PubMed for medical articles

        Args:
            query: PubMed search query
            max_results: Maximum number of articles to retrieve

        Returns:
            List of article dictionaries
        """
        print(f"🔍 Searching PubMed for: '{query}' (max: {max_results})")

        # PubMed E-utilities API
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

        # Search for articles
        search_url = f"{base_url}esearch.fcgi"
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': min(max_results, 1000),  # API limit is 1000 per request
            'retmode': 'json',
            'api_key': self.pubmed_api_key
        }

        try:
            response = requests.get(search_url, params=search_params, timeout=30)
            response.raise_for_status()
            search_data = response.json()

            pmids = search_data['esearchresult']['idlist']
            print(f"📄 Found {len(pmids)} article IDs")

            if not pmids:
                return []

            # Fetch article details in batches
            articles = []
            batch_size = 50

            for i in range(0, len(pmids), batch_size):
                batch_pmids = pmids[i:i + batch_size]

                # Fetch summaries
                summary_url = f"{base_url}esummary.fcgi"
                summary_params = {
                    'db': 'pubmed',
                    'id': ','.join(batch_pmids),
                    'retmode': 'json',
                    'api_key': self.pubmed_api_key
                }

                summary_response = requests.get(summary_url, params=summary_params, timeout=30)
                summary_data = summary_response.json()

                # Fetch abstracts
                fetch_url = f"{base_url}efetch.fcgi"
                fetch_params = {
                    'db': 'pubmed',
                    'id': ','.join(batch_pmids),
                    'retmode': 'xml',
                    'rettype': 'abstract',
                    'api_key': self.pubmed_api_key
                }

                fetch_response = requests.get(fetch_url, params=fetch_params, timeout=30)
                fetch_data = fetch_response.text

                # Parse XML for abstracts
                root = ET.fromstring(fetch_data)

                for pmid in batch_pmids:
                    try:
                        article = root.find(f".//PubmedArticle[MedlineCitation/PMID='{pmid}']")
                        if article is None:
                            continue

                        # Extract basic info
                        medline_citation = article.find('MedlineCitation')
                        pmid_elem = medline_citation.find('PMID')
                        article_elem = medline_citation.find('Article')

                        title_elem = article_elem.find('ArticleTitle')
                        title = title_elem.text if title_elem is not None else "Unknown Title"

                        # Extract abstract
                        abstract_elem = article_elem.find('.//Abstract/AbstractText')
                        abstract = ""
                        if abstract_elem is not None:
                            # Handle multi-part abstracts
                            abstract_parts = []
                            for part in article_elem.findall('.//Abstract/AbstractText'):
                                if part.text:
                                    abstract_parts.append(part.text.strip())
                            abstract = ' '.join(abstract_parts)

                        # Extract authors
                        authors = []
                        for author in article_elem.findall('.//AuthorList/Author'):
                            last_name = author.find('LastName')
                            fore_name = author.find('ForeName')
                            if last_name is not None and fore_name is not None:
                                authors.append(f"{fore_name.text} {last_name.text}")

                        # Extract journal and year
                        journal_elem = article_elem.find('Journal/Title')
                        journal = journal_elem.text if journal_elem is not None else "Unknown Journal"

                        year_elem = article_elem.find('Journal/JournalIssue/PubDate/Year')
                        year = year_elem.text if year_elem is not None else datetime.now().year

                        article_data = {
                            'pmid': pmid,
                            'title': title.strip(),
                            'abstract': abstract.strip(),
                            'authors': authors[:3],  # Limit to first 3 authors
                            'journal': journal,
                            'year': int(year),
                            'source': 'pubmed',
                            'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            'query': query,
                            'retrieved_date': datetime.now().isoformat()
                        }

                        # Only include articles with substantial abstracts
                        if len(abstract.strip()) > 200:
                            articles.append(article_data)

                    except Exception as e:
                        print(f"❌ Error parsing PMID {pmid}: {e}")
                        continue

                # Rate limiting
                time.sleep(0.5)

            print(f"✅ Retrieved {len(articles)} articles with abstracts")
            return articles

        except Exception as e:
            print(f"❌ PubMed search failed: {e}")
            return []

    def scrape_clinical_guidelines(self) -> List[Dict]:
        """
        Scrape clinical practice guidelines from authoritative sources

        Returns:
            List of guideline dictionaries
        """
        print("📋 Scraping clinical practice guidelines...")

        guidelines = []

        # American College of Physicians (ACP) guidelines
        try:
            acp_guidelines = self._scrape_acp_guidelines()
            guidelines.extend(acp_guidelines)
        except Exception as e:
            print(f"❌ ACP scraping failed: {e}")

        # American Academy of Family Physicians (AAFP)
        try:
            aafp_guidelines = self._scrape_aafp_guidelines()
            guidelines.extend(aafp_guidelines)
        except Exception as e:
            print(f"❌ AAFP scraping failed: {e}")

        # Centers for Disease Control and Prevention (CDC)
        try:
            cdc_guidelines = self._scrape_cdc_guidelines()
            guidelines.extend(cdc_guidelines)
        except Exception as e:
            print(f"❌ CDC scraping failed: {e}")

        print(f"✅ Retrieved {len(guidelines)} clinical guidelines")
        return guidelines

    def _scrape_acp_guidelines(self) -> List[Dict]:
        """Scrape ACP clinical guidelines"""
        guidelines = []

        # ACP maintains a guidelines page
        url = "https://www.acponline.org/clinical-information/guidelines"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Look for guideline links and content
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'guideline' in href.lower() or 'recommendation' in href.lower():
                    title = link.get_text().strip()
                    if title and len(title) > 10:
                        guidelines.append({
                            'title': title,
                            'url': href if href.startswith('http') else f"https://www.acponline.org{href}",
                            'source': 'ACP',
                            'retrieved_date': datetime.now().isoformat(),
                            'type': 'clinical_guideline'
                        })

        except Exception as e:
            print(f"❌ ACP scraping error: {e}")

        return guidelines

    def _scrape_aafp_guidelines(self) -> List[Dict]:
        """Scrape AAFP clinical guidelines"""
        guidelines = []

        # AAFP clinical practice guidelines
        url = "https://www.aafp.org/family-physician/patient-care/clinical-recommendations.html"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Look for recommendation links
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'recommendation' in href.lower():
                    title = link.get_text().strip()
                    if title and len(title) > 10:
                        guidelines.append({
                            'title': title,
                            'url': href if href.startswith('http') else f"https://www.aafp.org{href}",
                            'source': 'AAFP',
                            'retrieved_date': datetime.now().isoformat(),
                            'type': 'clinical_guideline'
                        })

        except Exception as e:
            print(f"❌ AAFP scraping error: {e}")

        return guidelines

    def _scrape_cdc_guidelines(self) -> List[Dict]:
        """Scrape CDC clinical guidelines"""
        guidelines = []

        # CDC clinical guidelines
        url = "https://www.cdc.gov/clinical-guidelines.html"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Look for guideline content
            for section in soup.find_all(['h2', 'h3', 'h4']):
                title = section.get_text().strip()
                if 'guideline' in title.lower() or 'recommendation' in title.lower():
                    guidelines.append({
                        'title': title,
                        'url': "https://www.cdc.gov/clinical-guidelines.html",
                        'source': 'CDC',
                        'retrieved_date': datetime.now().isoformat(),
                        'type': 'clinical_guideline'
                    })

        except Exception as e:
            print(f"❌ CDC scraping error: {e}")

        return guidelines

    def scrape_medical_journals(self) -> List[Dict]:
        """
        Scrape recent articles from major medical journals

        Returns:
            List of journal article dictionaries
        """
        print("📚 Scraping medical journal articles...")

        articles = []

        # Major medical journals with RSS feeds or APIs
        journals = {
            'NEJM': 'https://www.nejm.org/medical-articles?query=&sort=date',
            'JAMA': 'https://jamanetwork.com/journals/jama/issues',
            'Lancet': 'https://www.thelancet.com/journals/lancet/current',
            'BMJ': 'https://www.bmj.com/content/current'
        }

        for journal_name, url in journals.items():
            try:
                journal_articles = self._scrape_journal_articles(journal_name, url)
                articles.extend(journal_articles)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"❌ {journal_name} scraping failed: {e}")

        print(f"✅ Retrieved {len(articles)} journal articles")
        return articles

    def _scrape_journal_articles(self, journal_name: str, url: str) -> List[Dict]:
        """Scrape articles from a specific journal"""
        articles = []

        try:
            response = requests.get(url, timeout=30, headers={'User-Agent': 'MedicalDataIngester/1.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract article information (this is journal-specific)
            for article in soup.find_all(['article', 'div'], class_=re.compile(r'article|paper|content')):
                title_elem = article.find(['h1', 'h2', 'h3', 'a'])
                if title_elem:
                    title = title_elem.get_text().strip()
                    if len(title) > 20:  # Filter short titles
                        articles.append({
                            'title': title,
                            'journal': journal_name,
                            'url': url,
                            'source': journal_name,
                            'retrieved_date': datetime.now().isoformat(),
                            'type': 'journal_article'
                        })

        except Exception as e:
            print(f"❌ {journal_name} scraping error: {e}")

        return articles

    def chunk_medical_text(self, text: str, source_type: str = "general") -> List[str]:
        """
        Intelligent chunking for medical documents

        Args:
            text: Medical text to chunk
            source_type: Type of medical content (guideline, journal, drug_info)

        Returns:
            List of text chunks optimized for medical context
        """
        chunks = []

        # Medical-specific chunking strategies
        if source_type == "guideline":
            # Guidelines need context preservation - use section-based chunking
            chunks = self._chunk_guidelines(text)
        elif source_type == "journal":
            # Journal articles - preserve abstract, methods, results, conclusion
            chunks = self._chunk_journal_article(text)
        elif source_type == "drug_info":
            # Drug information - chunk by drug sections, indications, contraindications
            chunks = self._chunk_drug_info(text)
        else:
            # General medical text - use paragraph-aware chunking
            chunks = self._chunk_general_medical(text)

        # Filter out very small chunks
        chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 200]

        print(f"📦 Created {len(chunks)} medical chunks from {source_type}")
        return chunks

    def _chunk_guidelines(self, text: str) -> List[str]:
        """Chunk clinical guidelines preserving section structure"""
        chunks = []

        # Split by common guideline section markers
        sections = re.split(r'\n\s*(?:RECOMMENDATION|GUIDELINE|SUGGESTION|SECTION)\s*\d*\.?:?\s*\n', text, flags=re.IGNORECASE)

        current_chunk = ""
        for section in sections:
            section = section.strip()
            if not section:
                continue

            # If adding this section would exceed chunk size, save current and start new
            if len(current_chunk) + len(section) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                current_chunk = current_chunk[-self.chunk_overlap:] + "\n\n" + section
            else:
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _chunk_journal_article(self, text: str) -> List[str]:
        """Chunk journal articles preserving structure"""
        chunks = []

        # Try to identify and preserve key sections
        sections = {
            'abstract': re.search(r'\n\s*ABSTRACT\s*\n(.*?)(?:\n\s*(?:INTRODUCTION|METHODS|RESULTS|CONCLUSION|DISCUSSION)\s*\n|$)', text, re.DOTALL | re.IGNORECASE),
            'introduction': re.search(r'\n\s*INTRODUCTION\s*\n(.*?)(?:\n\s*(?:METHODS|RESULTS|CONCLUSION|DISCUSSION)\s*\n|$)', text, re.DOTALL | re.IGNORECASE),
            'methods': re.search(r'\n\s*MATERIALS?\s+AND\s+METHODS?\s*\n(.*?)(?:\n\s*(?:RESULTS|CONCLUSION|DISCUSSION)\s*\n|$)', text, re.DOTALL | re.IGNORECASE),
            'results': re.search(r'\n\s*RESULTS?\s*\n(.*?)(?:\n\s*(?:CONCLUSION|DISCUSSION)\s*\n|$)', text, re.DOTALL | re.IGNORECASE),
            'discussion': re.search(r'\n\s*DISCUSSION\s*\n(.*?)(?:\n\s*CONCLUSION\s*\n|$)', text, re.DOTALL | re.IGNORECASE),
            'conclusion': re.search(r'\n\s*CONCLUSION\s*\n(.*?)$', text, re.DOTALL | re.IGNORECASE)
        }

        # Extract and chunk each section
        for section_name, match in sections.items():
            if match:
                section_text = match.group(1).strip()
                if len(section_text) > 100:
                    # Chunk large sections
                    if len(section_text) > self.chunk_size:
                        for i in range(0, len(section_text), self.chunk_size - self.chunk_overlap):
                            chunk = section_text[i:i + self.chunk_size].strip()
                            if len(chunk) > 100:
                                chunks.append(f"[{section_name.upper()}]\n{chunk}")
                    else:
                        chunks.append(f"[{section_name.upper()}]\n{section_text}")

        # If no structured sections found, fall back to general chunking
        if not chunks:
            chunks = self._chunk_general_medical(text)

        return chunks

    def _chunk_drug_info(self, text: str) -> List[str]:
        """Chunk drug information by sections"""
        chunks = []

        # Split by drug information sections
        sections = re.split(r'\n\s*(?:INDICATIONS?|CONTRAINDICATIONS?|WARNINGS?|ADVERSE\s+REACTIONS?|DOSAGE?|DRUG\s+INTERACTIONS?)\s*:?\s*\n', text, flags=re.IGNORECASE)

        current_chunk = ""
        for section in sections:
            section = section.strip()
            if not section:
                continue

            # If adding this section would exceed chunk size, save current and start new
            if len(current_chunk) + len(section) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = current_chunk[-self.chunk_overlap:] + "\n\n" + section
            else:
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _chunk_general_medical(self, text: str) -> List[str]:
        """General medical text chunking with paragraph awareness"""
        chunks = []

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        current_chunk = ""
        for para in paragraphs:
            # If adding this paragraph would exceed chunk size, save current and start new
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                current_chunk = current_chunk[-self.chunk_overlap:] + "\n\n" + para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def save_medical_documents(self, documents: List[Dict], source_type: str):
        """
        Save medical documents to appropriate directories

        Args:
            documents: List of document dictionaries
            source_type: Type of documents (pubmed, guidelines, journals, etc.)
        """
        print(f"💾 Saving {len(documents)} {source_type} documents...")

        saved_count = 0

        for doc in documents:
            try:
                # Create filename from title or ID
                if 'pmid' in doc:
                    filename = f"pmid_{doc['pmid']}.json"
                elif 'title' in doc:
                    # Create safe filename from title
                    safe_title = re.sub(r'[^\w\s-]', '', doc['title'][:50]).strip().replace(' ', '_')
                    filename = f"{safe_title}_{int(time.time())}.json"
                else:
                    filename = f"doc_{int(time.time())}_{saved_count}.json"

                # Save to appropriate directory
                if source_type == "pubmed":
                    filepath = self.sources_dir / filename
                elif source_type == "guidelines":
                    filepath = self.guidelines_dir / filename
                elif source_type == "journals":
                    filepath = self.journals_dir / filename
                else:
                    filepath = self.medical_dir / filename

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(doc, f, indent=2, ensure_ascii=False)

                saved_count += 1

                # Update state
                doc_id = doc.get('pmid', doc.get('title', filename))
                self.state["sources_processed"][doc_id] = {
                    "filepath": str(filepath),
                    "processed_date": datetime.now().isoformat(),
                    "source_type": source_type
                }

            except Exception as e:
                print(f"❌ Error saving document: {e}")

        print(f"✅ Saved {saved_count}/{len(documents)} documents")

    def rebuild_medical_embeddings(self):
        """
        Rebuild RAG embeddings specifically for medical content
        Uses medical-optimized chunking and indexing
        """
        print("🔄 Rebuilding medical embeddings...")

        # Collect all medical text files
        medical_files = []
        medical_files.extend(self.sources_dir.glob("*.json"))
        medical_files.extend(self.guidelines_dir.glob("*.json"))
        medical_files.extend(self.journals_dir.glob("*.json"))

        if not medical_files:
            print("❌ No medical documents found for embedding")
            return False

        print(f"📄 Found {len(medical_files)} medical documents")

        # Extract text from all medical documents
        all_texts = []
        metadata = []

        for filepath in medical_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    doc = json.load(f)

                # Extract text content based on document type
                text_content = self._extract_text_from_medical_doc(doc)
                if text_content and len(text_content.strip()) > 100:
                    all_texts.append(text_content)

                    # Store metadata for reference
                    metadata.append({
                        'filepath': str(filepath),
                        'title': doc.get('title', 'Unknown'),
                        'source': doc.get('source', 'Unknown'),
                        'type': doc.get('type', 'medical_document')
                    })

            except Exception as e:
                print(f"❌ Error processing {filepath}: {e}")

        if not all_texts:
            print("❌ No valid text content found")
            return False

        print(f"📝 Extracted {len(all_texts)} medical texts")

        # Create medical-specific chunks
        all_chunks = []
        chunk_metadata = []

        for i, text in enumerate(all_texts):
            chunks = self.chunk_medical_text(text, metadata[i]['type'])
            all_chunks.extend(chunks)

            # Track which document each chunk came from
            for chunk in chunks:
                chunk_metadata.append({
                    'document_index': i,
                    'source_file': metadata[i]['filepath'],
                    'title': metadata[i]['title'],
                    'source': metadata[i]['source']
                })

        print(f"📦 Created {len(all_chunks)} medical chunks")

        # Generate embeddings using medical-optimized model if available
        try:
            # Try to use a medical-specific model first
            model_name = "pritamdeka/BioBERT-mnli-snli-scinli-stsb"
            print(f"🧠 Loading medical model: {model_name}")

            encoder = SentenceTransformer(model_name, device='cuda' if 'cuda' in str(torch.device('cuda:0')) else 'cpu')
        except:
            # Fall back to general model
            print("🧠 Using general model: all-MiniLM-L6-v2")
            encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if 'cuda' in str(torch.device('cuda:0')) else 'cpu')

        print("🔢 Generating medical embeddings...")
        embeddings = encoder.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True)
        embeddings = embeddings.astype(np.float32)

        # Create FAISS index for medical content
        dimension = embeddings.shape[1]
        medical_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        medical_index.add(embeddings)

        # Save medical-specific embeddings
        medical_embeddings_dir = self.medical_dir / "embeddings"
        medical_embeddings_dir.mkdir(exist_ok=True)

        # Save index, vectors, chunks, and metadata
        faiss.write_index(medical_index, str(medical_embeddings_dir / "medical_index.faiss"))
        np.save(medical_embeddings_dir / "medical_vectors.npy", embeddings.astype(np.float32))
        np.save(medical_embeddings_dir / "medical_chunks.npy", np.array(all_chunks))
        json.dump(chunk_metadata, open(medical_embeddings_dir / "medical_metadata.json", 'w'))

        print("✅ Medical embeddings rebuilt successfully!")
        print(f"📊 Index: {medical_index.ntotal} vectors, dimension: {dimension}")
        print(f"📦 Chunks: {len(all_chunks)} medical text chunks")

        # Update state
        self.state["last_embedding_rebuild"] = datetime.now().isoformat()
        self.state["total_chunks"] = len(all_chunks)
        self.save_state()

        return True

    def _extract_text_from_medical_doc(self, doc: Dict) -> str:
        """Extract text content from medical document"""
        text_parts = []

        # Extract based on document structure
        if 'abstract' in doc and doc['abstract']:
            text_parts.append(f"ABSTRACT: {doc['abstract']}")

        if 'title' in doc and doc['title']:
            text_parts.append(f"TITLE: {doc['title']}")

        if 'content' in doc and doc['content']:
            text_parts.append(doc['content'])

        # For guidelines and journal articles, try to extract main content
        if 'summary' in doc and doc['summary']:
            text_parts.append(f"SUMMARY: {doc['summary']}")

        if 'recommendations' in doc and doc['recommendations']:
            text_parts.append(f"RECOMMENDATIONS: {doc['recommendations']}")

        # Add metadata
        metadata_parts = []
        if 'authors' in doc and doc['authors']:
            metadata_parts.append(f"Authors: {', '.join(doc['authors'])}")

        if 'journal' in doc and doc['journal']:
            metadata_parts.append(f"Journal: {doc['journal']}")

        if 'year' in doc:
            metadata_parts.append(f"Year: {doc['year']}")

        if metadata_parts:
            text_parts.append(f"METADATA: {', '.join(metadata_parts)}")

        return '\n\n'.join(text_parts)

    def run_full_ingestion(self, force_update: bool = False):
        """
        Run complete medical data ingestion pipeline

        Args:
            force_update: Force update even if recently updated
        """
        print("🚀 Starting full medical data ingestion...")

        # Check if we should skip (unless forced)
        if not force_update and self.state.get("last_update"):
            last_update = datetime.fromisoformat(self.state["last_update"])
            if datetime.now() - last_update < timedelta(hours=24):
                print("⏭️ Skipping ingestion (updated recently)")
                return True

        # Step 1: Scrape PubMed articles
        print("\n📄 Step 1: Scraping PubMed articles...")
        pubmed_articles = []

        # Common medical queries for comprehensive coverage
        medical_queries = [
            "cardiology clinical trials 2024",
            "diabetes management guidelines",
            "infectious disease treatment",
            "neurology recent advances",
            "oncology immunotherapy",
            "pediatrics vaccination",
            "psychiatry mental health",
            "pulmonology respiratory",
            "emergency medicine protocols",
            "family medicine guidelines"
        ]

        for query in medical_queries:
            articles = self.scrape_pubmed_articles(query, max_results=50)
            pubmed_articles.extend(articles)
            time.sleep(1)  # Rate limiting

        self.save_medical_documents(pubmed_articles, "pubmed")

        # Step 2: Scrape clinical guidelines
        print("\n📋 Step 2: Scraping clinical guidelines...")
        guidelines = self.scrape_clinical_guidelines()
        self.save_medical_documents(guidelines, "guidelines")

        # Step 3: Scrape medical journals
        print("\n📚 Step 3: Scraping medical journals...")
        journal_articles = self.scrape_medical_journals()
        self.save_medical_documents(journal_articles, "journals")

        # Step 4: Rebuild embeddings
        print("\n🔄 Step 4: Rebuilding medical embeddings...")
        success = self.rebuild_medical_embeddings()

        if success:
            # Update state
            self.state["last_update"] = datetime.now().isoformat()
            self.state["total_articles"] = len(pubmed_articles)
            self.state["total_guidelines"] = len(guidelines)
            self.save_state()

            print("✅ Full medical data ingestion completed!")
        else:
            print("❌ Medical data ingestion failed!")

        return success

    def get_medical_stats(self) -> Dict:
        """Get statistics about medical data collection"""
        stats = {
            'total_documents': 0,
            'sources': {},
            'last_update': self.state.get('last_update'),
            'total_chunks': self.state.get('total_chunks', 0)
        }

        # Count documents by source
        for source_dir in [self.sources_dir, self.guidelines_dir, self.journals_dir]:
            if source_dir.exists():
                count = len(list(source_dir.glob("*.json")))
                stats['total_documents'] += count

                source_name = source_dir.name
                stats['sources'][source_name] = count

        return stats

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Medical Data Ingestion for RAG System')
    parser.add_argument('--scrape', action='store_true', help='Scrape medical data from sources')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild medical embeddings')
    parser.add_argument('--update', action='store_true', help='Run full update (scrape + rebuild)')
    parser.add_argument('--stats', action='store_true', help='Show medical data statistics')
    parser.add_argument('--force', action='store_true', help='Force update even if recently updated')

    args = parser.parse_args()

    ingester = MedicalDataIngester()

    if args.stats:
        stats = ingester.get_medical_stats()
        print("📊 Medical Data Statistics:")
        print(json.dumps(stats, indent=2))

    elif args.scrape:
        print("🔍 Scraping medical data...")
        # Just scrape, don't rebuild embeddings
        pubmed_articles = ingester.scrape_pubmed_articles("medicine", max_results=50)
        ingester.save_medical_documents(pubmed_articles, "pubmed")

        guidelines = ingester.scrape_clinical_guidelines()
        ingester.save_medical_documents(guidelines, "guidelines")

        print("✅ Scraping completed!")

    elif args.rebuild:
        print("🔄 Rebuilding medical embeddings...")
        success = ingester.rebuild_medical_embeddings()
        print("✅ Rebuild completed!" if success else "❌ Rebuild failed!")

    elif args.update:
        print("🚀 Running full medical data update...")
        success = ingester.run_full_ingestion(force_update=args.force)
        print("✅ Update completed!" if success else "❌ Update failed!")

    else:
        print("Usage: python3 medical_data_ingestion.py [--scrape|--rebuild|--update|--stats] [--force]")
        print("Use --update for full medical data ingestion pipeline")
