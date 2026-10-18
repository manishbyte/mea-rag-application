<<<<<<< HEAD
# MEA Annual Report 2025 — Secure RAG Chatbot

A Retrieval-Augmented Generation (RAG) application for answering questions about the Ministry of External Affairs (MEA), Government of India, Annual Report 2025. The system combines document retrieval, semantic search, keyword search, reranking, evidence-grounded generation, citation validation, security guardrails, and evaluation to make answers more verifiable and reliable.

## Overview

The MEA Annual Report 2025 contains information about India's foreign relations, diplomatic engagements, international partnerships, multilateral cooperation, consular services, and other foreign policy initiatives. Finding specific facts, statistics, and page references in a lengthy report can be time-consuming.

This project provides a conversational interface where users can ask natural-language questions and receive answers grounded in the report, with source citations and page references when available.

Unlike a general-purpose chatbot that may answer from its pretrained knowledge, this application is designed to retrieve relevant evidence from the indexed report, assess whether that evidence is sufficient, and avoid unsupported answers.

### Key objectives

- Answer questions using the indexed MEA Annual Report 2025.
- Retrieve relevant information through semantic and keyword search.
- Improve retrieval quality through Reciprocal Rank Fusion and cross-encoder reranking.
- Generate evidence-grounded answers using an LLM.
- Validate citations against retrieved evidence.
- Abstain when sufficient supporting evidence is unavailable.
- Protect the application against malicious inputs and prompt-injection attempts.
- Persist application data and conversation history in PostgreSQL.
- Evaluate retrieval, answer quality, citations, security behavior, and latency.
- Deploy the frontend and backend as separate services.

## Key Features

### 1. Retrieval-Augmented Generation

The application follows a retrieval-first approach. Instead of relying exclusively on an LLM's internal knowledge, it searches the indexed annual report and provides relevant passages as context for answer generation.

The workflow includes:

- Input validation and security checks.
- Query embedding generation.
- Vector-based document retrieval.
- PostgreSQL full-text keyword retrieval.
- Reciprocal Rank Fusion (RRF).
- Cross-encoder reranking.
- Evidence sufficiency assessment.
- Grounded answer generation.
- Citation validation and output checks.

### 2. Hybrid Search

The retrieval system combines complementary search strategies.

**Semantic search:** Uses vector embeddings to identify passages that are conceptually related to a question, even when the wording differs.

**Keyword search:** Uses PostgreSQL Full-Text Search to find relevant terms, names, phrases, and factual expressions.

**Reciprocal Rank Fusion:** Combines the ranked results from the retrieval methods and reduces dependence on either method alone.

**Cross-encoder reranking:** Re-evaluates candidate passages against the question to improve the relevance of the evidence supplied to the generation model.

This hybrid approach is intended to improve retrieval for both natural-language questions and queries involving specific names, dates, figures, and terminology.

### 3. Evidence-Grounded Answers and Citations

The generation workflow is instructed to use retrieved report content as evidence rather than treating it as a source of new instructions.

The application is designed to:

- Ground substantive claims in retrieved passages.
- Associate citations with source documents and page metadata.
- Validate citation references against available evidence.
- Avoid inventing page numbers or source references.
- Return an appropriate refusal or abstention when evidence is insufficient.

Citation validation improves traceability, but it does not automatically prove that every generated statement is factually correct. Evaluation and manual verification remain important.

### 4. Security Guardrails

Security controls are incorporated into the request and answer workflows.

**Input protection**
- Validate incoming requests and question lengths.
- Reject malformed or disallowed inputs.
- Detect suspicious prompt-injection attempts.
- Prevent retrieval and generation when a request fails the applicable input checks.

**Retrieval protection**
- Treat retrieved passages as untrusted evidence.
- Keep retrieved document text separate from system instructions.
- Validate source metadata and citation references.
- Avoid executing instructions, code, or queries found in document content.

**Output protection**
- Validate generated answers and citations.
- Check for unsupported answers and unsafe content.
- Avoid exposing internal prompts, credentials, API keys, or database secrets in responses.
- Return controlled errors rather than leaking internal stack traces.

These controls use defense in depth. No prompt-injection detector can guarantee that every attack will be blocked, so security testing and continuous evaluation are essential.

### 5. Persistent Conversations and Authentication

The application includes user authentication and persistent conversation management.

The backend supports:

- User registration and login.
- Token-based authentication.
- Password hashing.
- Authenticated chat requests.
- Conversation history retrieval.
- Persistent messages and citations.
- Conversation ownership checks.
- Logout handling in the client.

PostgreSQL is used to persist application records. Authentication and authorization checks help prevent users from accessing conversations that do not belong to them.

### 6. Evaluation Framework

The project includes a versioned evaluation dataset and an independent evaluation runner. Evaluation can be performed without making it part of the normal chat request path.

The evaluation framework is intended to measure several aspects of system quality.

| Category | Metrics and checks |
|---|---|
| Retrieval | Recall@k, MRR, recall of relevant pages |
| Answer quality | Token F1, ROUGE-L F1, exact match |
| Citations | Citation correctness and source/page checks |
| Security | Guardrail behavior, attack rejection, security test results |
| Operations | End-to-end latency and execution failures |

The available dashboard presents retrieval, answer-quality, citation, security, and latency results.

These metrics provide complementary signals. For example, high retrieval recall does not guarantee that an answer is correct, and text-overlap metrics do not fully measure semantic faithfulness.

Evaluation results should be generated by actually running the evaluation suite. No benchmark scores or acceptance-threshold results are claimed here.

## System Architecture

The application separates the user interface, API, retrieval workflow, database, and document ingestion responsibilities.

```text
User
 |
 v
Streamlit Frontend
 |
 | HTTPS / REST API
 v
FastAPI Backend
 |
 +--> Authentication and Authorization
 |
 +--> Request Validation and Security Guardrails
 |
 +--> LangGraph RAG Workflow
 |       |
 |       +--> Query Validation
 |       +--> Query Embedding
 |       +--> Vector Retrieval
 |       +--> PostgreSQL Full-Text Search
 |       +--> Reciprocal Rank Fusion
 |       +--> Cross-Encoder Reranking
 |       +--> Evidence Sufficiency Check
 |       +--> Grounded Answer Generation
 |       +--> Citation and Output Validation
 |
 +--> Neon PostgreSQL
         |
         +--> Users and Conversations
         +--> Messages and Citations
         +--> Document Metadata
         +--> Chunks and Embeddings
```

Document ingestion is a separate workflow. It extracts text from the source PDF, preserves page metadata, creates chunks, generates embeddings, and persists the indexed content. It is not intended to run as a full re-ingestion operation whenever the API starts.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Frontend | Streamlit |
| Backend | FastAPI |
| API validation | Pydantic |
| Workflow orchestration | LangGraph |
| LLM integration | LangChain |
| LLM inference | NVIDIA-hosted model through ChatNVIDIA |
| Starting generation model | `openai/gpt-oss-20b` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Relational database | Neon PostgreSQL |
| Vector search | PostgreSQL with pgvector |
| Keyword retrieval | PostgreSQL Full-Text Search |
| Asynchronous database access | SQLAlchemy and asyncpg |
| Authentication | JWT and Argon2-capable password hashing |
| Evaluation | Python-based evaluation runner and pytest |
| Hosting target | Render |
| Database hosting | Neon |

**Model provider note:** The current generation integration uses NVIDIA's LangChain integration. The original architecture considered Groq, but the README describes the NVIDIA-based implementation rather than claiming that Groq is currently used.

## Database and Persistence

Neon PostgreSQL acts as the persistent data store. The pgvector extension allows document embeddings to be stored and queried alongside relational data.

The database is intended to support:

- Document metadata and ingestion status.
- Page-aware document chunks.
- Vector embeddings.
- User accounts.
- Conversations and messages.
- Citation metadata.
- Evaluation records where persistence is implemented.

The embedding model produces 384-dimensional vectors, so the vector column must use a compatible dimension.

Enable pgvector in the target Neon database when it has not already been enabled:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Database schema changes should be managed through Alembic migrations rather than destructive schema resets. The indexed knowledge base should remain in durable storage across application restarts and redeployments.

## Prerequisites

Before running the application, install or configure:

- Python compatible with the project's dependencies and deployment runtime.
- Git.
- A Neon PostgreSQL database.
- An NVIDIA API key with access to the configured generation model.
- The official MEA Annual Report 2025 PDF.
- The dependencies listed in `requirements.txt`.

Model downloads and first-time model initialization may require additional time and disk space.

## Local Setup

The following commands are intended for Windows PowerShell.

### 1. Clone the repository

Replace the placeholder URL with your actual GitHub repository URL.

```powershell
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create and activate a virtual environment

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If Python 3.13 is not installed, use the Python version supported by the project's pinned dependencies and deployment configuration.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file from the example configuration:

```powershell
Copy-Item .env.example .env
```

Set the required values in `.env` using your own credentials and deployment settings.

Configuration typically includes:

- Neon PostgreSQL connection settings.
- NVIDIA API credentials and model configuration.
- JWT signing configuration.
- Backend runtime and logging settings.
- Retrieval and generation limits.
- Frontend API base URL.

Use the variable names already defined in `.env.example` and the application's configuration. Do not commit `.env` or paste real credentials into issues, logs, or public documentation.

### 5. Apply database migrations

If the repository has initialized Alembic and contains the required migrations, run:

```powershell
alembic upgrade head
```

Confirm that the migration configuration points to the intended database before applying schema changes.

### 6. Prepare the knowledge base

The official annual report must be ingested and verified before production questions can be answered reliably.

Run the document ingestion command provided by the implemented ingestion pipeline. Verify that:

- The PDF was parsed successfully.
- Page numbers were preserved.
- Chunks and embeddings were persisted.
- Vector dimensions match the configured model.
- Retrieval returns relevant passages.
- The knowledge base is ready for use.

Ingestion should be repeatable and should not create duplicate records when the same document and ingestion configuration are processed again.

### 7. Start the FastAPI backend

From the repository root:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Useful local endpoints include:

- `http://127.0.0.1:8000/health/live`
- `http://127.0.0.1:8000/health/ready`
- `http://127.0.0.1:8000/docs`

The readiness endpoint should be used to verify required dependencies, not merely that the web server process is running.

### 8. Start the Streamlit frontend

Open another PowerShell terminal, activate the same virtual environment, and run:

```powershell
python -m streamlit run ui/app.py
```

Configure the frontend API URL to point to the running FastAPI service. For local development, the usual base URL is:

```text
http://127.0.0.1:8000
```

Open the local Streamlit URL printed in the terminal, register or log in as required, and submit a question about the annual report.

## API Overview

The FastAPI backend exposes endpoints for authentication, chat, conversation history, and service health.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/auth/register` | Register a user |
| POST | `/auth/login` | Authenticate a user |
| GET | `/auth/me` | Retrieve the authenticated user's profile |
| POST | `/auth/logout` | Handle client logout |
| POST | `/chat` | Submit a question and receive an answer |
| GET | `/chat/conversations` | List the user's conversations |
| GET | `/chat/conversations/{conversation_id}` | Retrieve a conversation and its messages |
| GET | `/health/live` | Check whether the API process is alive |
| GET | `/health/ready` | Check application readiness |

Authentication requirements and request schemas are defined by the API implementation. Consult the interactive API documentation at `/docs` when the backend is running.

**Logout limitation:** If JWTs are stateless, the logout endpoint alone does not revoke a previously issued token. Secure client-side token deletion is required; immediate server-side revocation requires an implemented token-revocation mechanism.

## Running Evaluation

The evaluation runner is separate from the interactive chat interface.

From the repository root, run:

```powershell
python -m evaluation.run --dataset evaluation/datasets/mea_v1.jsonl
```

The dataset should contain representative questions and expected evidence, including factual, numerical, multi-page, ambiguous, unanswerable, citation-sensitive, and adversarial cases.

Review the evaluation summary and investigate failures rather than relying on a single aggregate score. Record the dataset, prompt, model, and configuration versions when comparing results between releases.

The evaluation suite should be run after material changes to retrieval, chunking, embeddings, prompts, model configuration, citation logic, or guardrails.

## Testing and Code Quality

Run the available test suite from the repository root:

```powershell
python -m pytest -q
```

Tests should cover:

- Request validation and security guardrails.
- Chunk metadata and page preservation.
- Embedding dimensions.
- Hybrid retrieval and RRF ranking.
- Citation construction and validation.
- Evidence sufficiency and abstention.
- Authentication and conversation ownership.
- Database persistence.
- API error handling.
- Evaluation metric calculations.
- Deployment readiness and configuration failures.

External model calls should be mocked in most unit tests. Real database and provider integration tests should be controlled and configured separately.

A successful test command should be reported as passing only after the tests have actually executed successfully.

## Security and Secret Management

Security is a core requirement of this project.

- Store API keys, database credentials, and JWT signing secrets in environment variables or the hosting platform's secret configuration.
- Keep `.env` excluded from Git.
- Use HTTPS for deployed frontend-to-backend communication.
- Configure CORS to allow only the required frontend origins.
- Validate inputs and apply request-size and rate limits.
- Enforce authentication and conversation ownership checks.
- Treat retrieved text as untrusted evidence.
- Validate citation references against actual retrieved material.
- Avoid exposing credentials, internal prompts, and stack traces.
- Keep ingestion and administrative operations restricted to authorized users.
- Monitor security failures without logging secrets or unnecessary personal data.

If a credential is accidentally committed or exposed, rotate it immediately. Removing it from the latest commit does not necessarily remove it from Git history.

## Deployment

The intended hosting setup separates the user interface, API, and persistent database.

**Frontend:** Streamlit web service on Render.

**Backend:** FastAPI web service on Render.

**Database:** Neon PostgreSQL with pgvector enabled.

The backend start command is:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

The Streamlit service should use its own start command, for example:

```bash
streamlit run ui/app.py --server.address 0.0.0.0 --server.port $PORT
```

Configure the services with the appropriate environment variables, including the Neon connection settings, model credentials, JWT configuration, and frontend API base URL.

Before declaring deployment complete, verify:

1. The backend starts successfully with production configuration.
2. The database connection and migrations work.
3. The readiness endpoint reflects the actual state of required dependencies.
4. The frontend can authenticate and communicate with the deployed backend.
5. A real question retrieves relevant evidence and returns verifiable citations.
6. Guardrails reject the tested malicious requests.
7. Conversation history survives a service restart.
8. The indexed knowledge base remains available after a redeployment.
9. The evaluation runner executes successfully against the configured dataset.
10. Secrets do not appear in source control, API responses, or logs.

Deployment should not trigger full document re-ingestion during ordinary API startup. Use a controlled ingestion command or separate job when the source document or indexing configuration changes.

## Limitations and Reliability

This project is designed to improve the reliability of document question answering, but it cannot guarantee that every answer is correct.

Potential limitations include:

- Extraction errors in complex tables, scanned pages, or unusual PDF layouts.
- Retrieval failures for rare terminology or questions requiring distant context.
- Incorrect or incomplete generated explanations despite relevant evidence.
- Imperfect prompt-injection detection.
- Model-provider rate limits, latency, and availability.
- Differences between offline evaluation and real user questions.

The correct response to insufficient evidence is to qualify or abstain rather than invent an answer. Regular regression testing, citation review, and evaluation are necessary to measure and improve reliability.

## Development Roadmap

The project follows an incremental engineering process:

1. Repository setup and configuration.
2. Database foundation and migrations.
3. Page-aware document ingestion and persistence.
4. Hybrid retrieval and reranking.
5. LangGraph workflow and grounded generation.
6. Input, retrieval, and output guardrails.
7. Evaluation dataset, metrics, and regression checks.
8. FastAPI endpoints and persistent conversation history.
9. Streamlit user interface.
10. Render deployment.
11. Production verification, recovery, and rollback testing.

Each stage should be verified before the next is considered complete. Implemented, tested, and deployed features should be documented separately.

## Contributing

Contributions should preserve the application's grounding, security, and evaluation requirements.

1. Create a branch for the change.
2. Keep configuration and credentials out of source control.
3. Add tests for new behavior and security-sensitive changes.
4. Run the relevant test suite before submitting changes.
5. Document configuration changes and migration requirements.

## License and Data Source

**Data source:** Ministry of External Affairs, Government of India, Annual Report 2025.

Use the official report and follow the source's applicable terms and attribution requirements. Add an explicit project software license before distributing the application under an open-source license.

---

**Project status:** This README documents the intended architecture and the current NVIDIA-based implementation context. It does not certify that the application has passed every evaluation, security test, deployment check, or production acceptance criterion. Those results should be recorded after verification.
=======
# mea-rag-application
>>>>>>> origin/main
