# Reciprocity Platform - System Architecture Evolution

## Table 1: System When We Cloned the Repos

| Objective | Components & Services | Role & Explanation |
|-----------|----------------------|-------------------|
| **1. User Registration** | | |
| | `reciprocity-backend` (NestJS) | REST API with JWT authentication |
| | PostgreSQL (`users` table) | User credentials with `onboarding_status` field |
| | Supabase Auth | Google OAuth social login |
| | `AuthContext.tsx` | Frontend auth state management |
| **2. Onboarding** | | |
| | `reciprocity-ai` (FastAPI :8000) | Python AI service for conversational onboarding |
| | `ConversationalOnboardingContext.tsx` | React context for chat-based flow |
| | `OnboardingChat.tsx` | Chat UI with message bubbles |
| | `slot_extraction.py` | Regex-based extraction of slots from messages |
| | `context_manager.py` | Conversation state and slot tracking |
| | `progressive_disclosure.py` | Role-specific question sequences |
| | Redis | Session persistence |
| | DynamoDB (LocalStack) | `user_profiles` table for Q&A storage |
| **3. Persona Generation** | | |
| | `persona_service.py` | Generate AI personas from Q&A |
| | `persona_processing.py` (Celery task) | Async persona generation |
| | OpenAI GPT-4.1-mini | LLM for profile summary generation |
| | `user_summaries` PostgreSQL table | Stores generated personas |
| **4. Matching** | | |
| | `matching_service.py` | Find matches using similarity |
| | `embedding_service.py` | Generate/store 768-dim embeddings |
| | SentenceTransformers (all-mpnet-base-v2) | Embedding model |
| | PostgreSQL + pgvector | Vector similarity search |
| | Cosine similarity @ 0.5-0.7 threshold | Match scoring |
| **5. Discovery** | | |
| | `DiscoverPage.tsx` | Browse/search interface with filters |
| | `/api/discover` endpoint | Filtered user listing |
| | `UserCard.tsx` | Profile preview cards |
| **6. Dashboard** | | |
| | `DashboardPage.tsx` | Personalized home page |
| | `MainLayout.tsx` | App shell with auth guards |
| | Match recommendations widget | Shows compatible users |
| **7. Infrastructure** | | |
| | `docker-compose.yml` | 5 containers: ai-service, celery, postgres, redis, localstack |
| | Celery + Redis | Async task queue |
| | Webhook endpoints | Backend notifications for matches |
| **8. Testing & QA** | | |
| | No test coverage | 0% test cases |
| | No test users validated | Pipeline untested end-to-end |
| **9. Documentation** | | |
| | `README.md` | Basic setup instructions |
| | `CODEX.md` | Code style guidelines |

---

## Table 2: System After Our Improvements (February 2026)

| Objective | Components & Services | Role & Explanation |
|-----------|----------------------|-------------------|
| **1. User Registration** | | |
| | `reciprocity-backend` (NestJS) | REST API with JWT authentication |
| | PostgreSQL (`users` table) | User credentials with `onboarding_status` field |
| | Supabase Auth | Google OAuth social login |
| | `AuthContext.tsx` | Frontend auth state management |
| | **AuthGuard on `/script` endpoint** | **NEW** - Prevented unauthenticated DB migrations |
| | **Null checks added** | **NEW** - Prevents crashes from missing nested properties |
| **2. Onboarding** | | |
| | `reciprocity-ai` (FastAPI :8000) | Python AI service for conversational onboarding |
| | `ConversationalOnboardingContext.tsx` | React context for chat-based flow |
| | `OnboardingChat.tsx` | Chat UI with message bubbles |
| | `slot_extraction.py` | **IMPROVED** - Added founder detection, correction handling |
| | `context_manager.py` | **IMPROVED** - Integrated LLM extraction with fallback |
| | `progressive_disclosure.py` | Role-specific question sequences |
| | **`llm_slot_extractor.py`** | **NEW** - GPT-4o-mini based extraction with true comprehension |
| | Redis | Session persistence |
| | DynamoDB (LocalStack) | `user_profiles` table for Q&A storage |
| **3. Persona Generation** | | |
| | `persona_service.py` | Generate AI personas from Q&A |
| | `persona_processing.py` (Celery task) | Async persona generation |
| | OpenAI GPT-4.1-mini | LLM for profile summary generation |
| | `user_summaries` PostgreSQL table | Stores generated personas |
| | **`postgresql_adapter.py`** | **FIXED** - `create_user_summary()` now called on completion |
| **4. Matching** | | |
| | `matching_service.py` | Find matches using similarity |
| | `embedding_service.py` | Generate/store 768-dim embeddings |
| | SentenceTransformers (all-mpnet-base-v2) | Embedding model |
| | PostgreSQL + pgvector | Vector similarity search |
| | Cosine similarity @ 0.5-0.7 threshold | Match scoring |
| | **`enhanced_matching_service.py`** | **NEW** - Bidirectional match scoring |
| | **Intent Classification** | **NEW** - investor_founder, mentor_mentee, cofounder detection |
| | **`feedback_learner.py`** | **NEW** - Embeddings adjust based on user feedback |
| | **Multi-Vector Embeddings** | **NEW** - Separate vectors for skills, industry, stage, culture |
| | **Match Explanation Engine** | **NEW** - Human-readable reasons for matches |
| | **Temporal + Activity Weighting** | **NEW** - Recent/active users ranked higher |
| | **Pre-computed Match Index** | **NEW** - Redis caching for faster lookups |
| **5. Discovery** | | |
| | `DiscoverPage.tsx` | Browse/search interface with filters |
| | `/api/discover` endpoint | Filtered user listing |
| | `UserCard.tsx` | Profile preview cards |
| **6. Dashboard** | | |
| | `DashboardPage.tsx` | Personalized home page |
| | `MainLayout.tsx` | App shell with auth guards |
| | Match recommendations widget | Shows compatible users |
| **7. Infrastructure** | | |
| | `docker-compose.yml` | 5 containers: ai-service, celery, postgres, redis, localstack |
| | Celery + Redis | Async task queue |
| | Webhook endpoints | Backend notifications for matches |
| | **257-line god method refactored** | **IMPROVED** - Split into 6 focused, testable methods |
| | **91 lines duplicated code extracted** | **IMPROVED** - Single reusable method |
| | **`isTheValueAnObject()` fix** | **FIXED** - Now actually validates JSON correctly |
| **8. Testing & QA** | | |
| | **37 new test cases** | **NEW** - Increased coverage from 0% |
| | **Batch 1: Alice, Bob, Charlie, Diana, Eve** | **NEW** - 5 users, all pipeline stages verified |
| | **Batch 2: Frank, Grace, Henry, Iris, Jack** | **NEW** - 5 users, cross-batch matching verified |
| | **Batch 3: Kevin, Laura, Mike, Nina, Oscar** | **NEW** - 5 users, 100% intent classification accuracy |
| | **15 total test users validated** | **NEW** - Full pipeline working end-to-end |
| | **`PARALLEL_ONBOARDING_TEST_USERS.md`** | **NEW** - 5 test personas for parallel QA testing |
| **9. Documentation** | | |
| | `README.md` | Basic setup instructions |
| | `CODEX.md` | Code style guidelines |
| | **`BACKEND_FIXES_SUMMARY.md`** | **NEW** - Security audit fixes documentation |
| | **`TEST_USERS_JOURNEY.md`** | **NEW** - Batch 1 test results |
| | **`TEST_USERS_BATCH2_JOURNEY.md`** | **NEW** - Batch 2 test results |
| | **`TEST_USERS_BATCH3_JOURNEY.md`** | **NEW** - Batch 3 and 7 AI improvements results |
| | **`SYSTEM_ARCHITECTURE_EVOLUTION.md`** | **NEW** - This document |
