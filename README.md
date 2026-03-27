# AI Legal Intelligence Platform

A comprehensive AI-powered platform for legal research and analysis focused on the Supreme Court of India, featuring citation verification, bail risk assessment, and case data management.

## Overview

This platform provides legal professionals with advanced AI tools for:
- **Citation Auditing**: Verify legal citations and quotes using Retrieval-Augmented Generation (RAG) with semantic search and LLM verification
- **Bail Reckoner**: AI-driven risk assessment for bail decisions based on legal precedents
- **Case Database**: Comprehensive archive of Supreme Court judgments with metadata extraction and search capabilities

## Features

### Citation Auditor
- Upload legal petitions (PDF) for citation analysis
- Extract and verify legal case citations against the Supreme Court database
- RAG-based quote verification using sentence embeddings and LLM inference
- Semantic search across judgment texts to find supporting or contradicting evidence
- Detailed verification reports with confidence scores

### Bail Reckoner
- Risk assessment engine for bail applications
- Considers factors like offense category, imprisonment served, and flight risk
- AI-powered recommendations based on legal precedents
- Interactive web interface for case evaluation

### Data Management
- Automated extraction of legal provisions from judgment texts
- Metadata enrichment for case records
- Parquet-based data storage for efficient querying
- Support for multiple years of Supreme Court data

## Architecture

### Backend
- **FastAPI**: High-performance web framework for API endpoints
- **Groq LLM**: For natural language processing and legal analysis
- **Sentence Transformers**: For semantic embedding generation
- **Pandas/PyArrow**: For data processing and parquet file management
- **PyPDF2**: For PDF text extraction

### Frontend
- **HTML/CSS/JavaScript**: Responsive web interface
- **Font Awesome**: Icon library for legal-themed UI
- **Custom CSS**: Dark theme with judicial color scheme

### Data Layer
- **Parquet Files**: Structured case metadata organized by year
- **PDF Archive**: Original judgment documents
- **SQLite/PostgreSQL**: Optional relational database for advanced queries

## Installation

### Prerequisites
- Python 3.8+
- Node.js (optional, for frontend development)
- Groq API key

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd legal
   ```

2. Install Python dependencies:
   ```bash
   pip install fastapi uvicorn python-dotenv groq sentence-transformers scikit-learn pandas pyarrow PyPDF2 python-multipart
   ```

3. Set up environment variables:
   Create a `.env` file with:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. Prepare data:
   - Place judgment PDFs in `judgments_[year]/extracted_[year]_cases/` directories
   - Run metadata extraction:
   ```bash
   python update_db.py
   ```

## Usage

### Starting the Server
```bash
python server.py
```
Access the web interface at `http://localhost:8000/app`

### API Endpoints

#### Citation Auditing
- `POST /audit`: Upload PDF for citation analysis
- `POST /chat`: Interactive citation verification
- `POST /verify-quote`: Direct quote verification

#### Bail Reckoner
- `POST /reckoner`: Bail risk assessment

#### Data Management
- `GET /cases`: Search case database
- `POST /update-metadata`: Update case metadata

### Web Interface
- **Citation Auditor**: Upload petitions and review verification results
- **Bail Reckoner**: Input case parameters for risk assessment
- **Dashboard**: Overview of system status and recent analyses

## Data Structure

### Judgment Organization
```
judgments_[year]/
├── english/
│   ├── english.index.json
│   └── extracted_[year]_cases/
│       ├── [case_id].pdf
│       └── ...
└── regional/
    ├── regional.index.json
    └── extracted_[year]_cases/
        ├── [case_id].pdf
        └── ...
```

### Metadata Format
Parquet files contain:
- Case ID and citation information
- Petitioner/respondent details
- Court and bench information
- Judgment date and year
- Legal provisions referenced
- Case descriptions and summaries

## Configuration

### Environment Variables
- `GROQ_API_KEY`: Required for LLM inference
- Database connection strings (if using external DB)

### Model Configuration
- Embedding model: `all-MiniLM-L6-v2`
- LLM: `llama-3.3-70b-versatile`
- Similarity threshold: 0.25 for quote verification

## Development

### Project Structure
```
legal/
├── main.py              # FastAPI application and core logic
├── server.py            # Server startup script
├── run.py               # Alternative server runner
├── update_db.py         # Metadata extraction and updates
├── verify_db.py         # Database verification utilities
├── detector.py          # Citation detection algorithms
├── miner.py             # Data mining utilities
├── explore.py           # Data exploration tools
├── walk.py              # File system utilities
├── frontend/
│   ├── templates/
│   │   ├── index.html
│   │   ├── auditor.html
│   │   └── bail-reckoner.html
│   ├── css/
│   │   └── styles.css
│   ├── js/
│   │   ├── main.js
│   │   ├── citation-auditor.js
│   │   └── bail-reckoner.js
│   └── static/
├── judgments_[year]/    # Case data by year
├── metadata_parquet/    # Processed metadata
└── .env                 # Environment configuration
```

### Adding New Features
1. Extend Pydantic models in `main.py` for new API endpoints
2. Add frontend templates in `frontend/templates/`
3. Implement JavaScript handlers in `frontend/js/`
4. Update data processing logic as needed

## Legal and Ethical Considerations

This platform is designed for legal research and analysis purposes. Users should:
- Verify all AI-generated outputs against primary legal sources
- Understand the limitations of AI in legal interpretation
- Use the platform as a research assistant, not a substitute for legal expertise
- Respect copyright and data usage rights for judgment texts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper documentation
4. Test thoroughly with sample legal documents
5. Submit a pull request

## License

[Specify your license here]

## Support

For technical support or feature requests, please [contact method].

## Version History

- v2.1: Enhanced citation verification with improved RAG
- v2.0: Added Bail Reckoner functionality
- v1.0: Initial Citation Auditor release