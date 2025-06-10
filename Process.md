# Building the Mahabharat RAG System: A Journey from Scratch to Deployment

This document chronicles the end-to-end process of creating a Retrieval-Augmented Generation (RAG) system for the Mahabharat, one of the longest and most revered epic poems in the world. The goal was to build a complete project from ideation to deployment, showcasing a methodical approach to data processing, system design, rigorous evaluation, and final implementation.

You can chat with the final result here: **[Mahabharat RAG Chatbot](https://mahabharat-rag.vercel.app/)**

---

## Table of Contents

1.  [**Phase 1: The Spark & The Setup**](#phase-1-the-spark--the-setup)
    *   [The Idea: Why the Mahabharat?](#the-idea-why-the-mahabharat)
    *   [Data Preparation & Intelligent Chunking](#data-preparation--intelligent-chunking)
2.  [**Phase 2: The First Attempt & The Reality Check**](#phase-2-the-first-attempt--the-reality-check)
    *   [Building a "Standard" RAG Pipeline](#building-a-standard-rag-pipeline)
    *   [Evaluation Woes: When Standard Metrics Fail](#evaluation-woes-when-standard-metrics-fail)
3.  [**Phase 3: The Pivot & The Methodical Rebuild**](#phase-3-the-pivot--the-methodical-rebuild)
    *   [A New Philosophy: Control and Context](#a-new-philosophy-control-and-context)
    *   [Crafting a High-Quality Evaluation Dataset](#crafting-a-high-quality-evaluation-dataset)
    *   [The Experimental Framework](#the-experimental-framework)
4.  [**Phase 4: Rigorous Experimentation & Results**](#phase-4-rigorous-experimentation--results)
    *   [Experiment A: The Naive RAG Baseline](#experiment-a-the-naive-rag-baseline)
    *   [Experiment B: Contextual Summaries - The Game Changer](#experiment-b-contextual-summaries---the-game-changer)
    *   [Experiment C: Hybrid Search with BM25](#experiment-c-hybrid-search-with-bm25)
    *   [Experiment D: The Reranking Hurdle](#experiment-d-the-reranking-hurdle)
5.  [**Phase 5: Final Architecture & Deployment**](#phase-5-final-architecture--deployment)
    *   [The Winning Pipeline](#the-winning-pipeline)
    *   [Deployment](#deployment)
6.  [**Future Work & Product-Level Enhancements**](#future-work--product-level-enhancements)
7.  [**Conclusion & Connect**](#conclusion--connect)

---

## Phase 1: The Spark & The Setup

### The Idea: Why the Mahabharat?

Every great project starts with a spark of inspiration. While scrolling through the [Vedabase.io](http://vedabase.io) platform, the idea struck: why not use the rich, complex, and deeply structured narrative of the Mahabharat? It presented a perfect challenge: a vast text with repeating characters, intricate plots, and profound philosophical discussions. This was an ideal candidate for a project that would demonstrate the ability to handle a complex problem from start to finish.

### Data Preparation & Intelligent Chunking

The source text was already neatly sorted by chapter, which was a great starting point. However, a naive chunking strategy wouldn't work.

**The Challenge:** Paragraphs varied wildly in length, from multi-line descriptions to single-line dialogues. Splitting chunks in the middle of a paragraph would break the context, which is fatal for a RAG system.

**The Strategy:**

1.  **Merge First, Then Chunk:** I decided to merge consecutive paragraphs together.
2.  **Set a Character Limit:** Chunks would be formed by adding paragraphs until the total character count approached a `1024` character limit, without exceeding it.
3.  **The "Why 1024?" Rationale:**
    *   A statistical analysis showed the 90th percentile for paragraph length was around 707 characters.
    *   `1024` was the nearest "pretty number" that provided a healthy buffer.
    *   This size translates to roughly 500-1000+ tokens (Devanagari script can be token-heavy), which is well within the context window of modern embedding models.

**The Process:**
I downloaded each chapter into a separate `.md` file and ran a script to implement this intelligent chunking logic, resulting in a clean, context-aware dataset.

## Phase 2: The First Attempt & The Reality Check

### Building a "Standard" RAG Pipeline

My first iteration used a fairly standard, off-the-shelf RAG architecture.

**The V1 Architecture Flowchart:**

```
[User Query]
     |
     v
[Sarvam AI Query Rewriter] -> (Reformulates & splits multi-part questions)
     |
     v
[Rewritten Query] -> (Embedded with Nomic)
     |
     v
[Qdrant Vector DB] -> (Similarity Search)
     |
     v
[Retrieved Context]
     |
     v
[Gemini 2.0 Flash] -> (Generates Answer)
     |
     v
[Final Answer]
```

This pipeline included a sophisticated query rewriter from Sarvam AI to handle complex user inputs.

### Evaluation Woes: When Standard Metrics Fail

To evaluate the system, I turned to **RAGAs**. However, the results were underwhelming and didn't seem to reflect the actual quality of the responses. As a sanity check, I manually evaluated 30 sample questions and answers using ChatGPT, and the results were quite satisfying.

**The Lesson:** Automated evaluation frameworks are powerful, but they aren't infallible. The discrepancy signaled that I either needed to fine-tune the evaluation or that the RAG pipeline itself, despite its standard components, wasn't optimized for the unique challenges of the Mahabharat text. This led to a major pivot.

## Phase 3: The Pivot & The Methodical Rebuild

### A New Philosophy: Control and Context

Dissatisfied with the initial results and inspired by the clarity of the **Anthropic RAG cookbook**, I decided to start fresh with a new philosophy:

1.  **Full Control:** Ditch external dependencies like Qdrant and complex pre-processing. I would manage embeddings in-memory and use `numpy` for similarity search. This gave me complete control and transparency.
2.  **Context is King:** The Mahabharat is dense with recurring names and keywords (Arjuna, Krishna, Dharma, etc.). Standard embeddings could easily get confused. The new hypothesis was that **enriching chunks with contextual information** before embedding would significantly improve retrieval accuracy.
3.  **Choosing the Right Tools:** This pivot also included a re-evaluation of the core components. While `nomic-embed-text` was used in the initial attempt, I switched to **Voyage AI's embedding models** for the rebuild. Their high performance and ability to handle nuanced semantic meaning made them a better fit for this refined, context-heavy approach.

### Crafting a High-Quality Evaluation Dataset

A good RAG system is built on good evaluation. This time, I took a more careful approach to creating the test set.

*   **Initial Failure:** My first attempt involved feeding the entire 500k-token book to Gemini 2.5 Pro, which crashed the model. Gemini 2.5 Flash worked but hallucinated answers due to the immense context.
*   **The Successful Approach:** I generated questions **chapter by chapter**. I prompted Gemini with the text of a single chapter and asked it to create complex, multi-hop questions, along with the correct answer and the exact context from the text.
*   **The Result:** A robust evaluation dataset of **876 question-answer-context triplets** in a structured JSON format.

### The Experimental Framework

With a solid dataset, I designed a series of experiments to methodically test different retrieval strategies. My key metric was **`Pass@N`**: *what percentage of the time are all the necessary ground-truth chunks for a question found within the top N retrieved documents?* A higher `Pass@N` means a better retriever.

**Evaluation Flowchart:**

```
[Evaluation Questions (876 total)]
           |
           v
+---------------------------+
|    RAG Retrieval System   |  <-- This is the component we'll swap
| (e.g., Naive, Summary, BM25)|
+---------------------------+
           |
           v
[Retrieved Chunks (Top N)]
           |
           v
+---------------------------+
|  Custom Metric: Pass@N    |
| (Checks if all ground-truth|
|  chunks are present)      |
+---------------------------+
           |
           v
[Final Pass@N Score]
```

## Phase 4: Rigorous Experimentation & Results

### Experiment A: The Naive RAG Baseline

This was the simplest approach: embedding the chunks directly and performing a similarity search.

*   **Results:**
    *   **Pass@5:** `69.52%`
    *   **Pass@10:** `77.64%`
    *   **Pass@20:** `84.41%`

This set a solid, but not spectacular, baseline to beat.

### Experiment B: Contextual Summaries - The Game Changer

Here, I tested the hypothesis that adding context would help. For each of the **2,897 chunks**, I used the Anthropic API to generate a concise summary. This summary was prepended to the chunk text before creating the embedding.

*   **The Grind:** This was a computationally intensive task, taking over **8 hours** to complete.
*   **The Payoff:** The results showed a massive improvement across the board.

*   **Results:**
    *   **Pass@5:** `78.43%` (**+8.9% improvement**)
    *   **Pass@10:** `87.05%` (**+9.4% improvement**)
    *   **Pass@20:** `92.15%` (**+7.7% improvement**)

**Conclusion:** The hypothesis was correct. The summaries provided crucial disambiguating context, allowing the embedding model to create much more effective representations of the chunks.

### Experiment C: Hybrid Search with BM25

Next, I tried a hybrid search approach, combining the "dense" vector search (from Experiment B) with a "sparse" keyword-based search (BM25, implemented with Elasticsearch).

*   **Results:**
    *   **Pass@5:** `78.42%` (No improvement)
    *   **Pass@10:** `87.67%` (+0.6% improvement)
    *   **Pass@20:** `92.81%` (+0.7% improvement)

**Conclusion:** The gains were negligible. This suggests that for this specific dataset and question set, the contextual summaries already provided enough semantic signal that keyword matching added little extra value.

### Experiment D: The Reranking Hurdle

The final experiment was to add a reranker model to re-order the retrieved results.

*   **The Challenge:** Production-grade rerankers are costly, and running local models without a GPU is prohibitively slow and often less accurate.
*   **The Decision:** After testing on a few samples and considering the practical constraints, I decided to **drop the reranking step**. The cost-benefit trade-off wasn't worth it for this project.

## Phase 5: Final Architecture & Deployment

### The Winning Pipeline

Based on the experimental results, the final architecture is a testament to the power of effective data enrichment over complex, multi-stage pipelines.

**Final Architecture Flowchart:**

```
                                  +----------------------------------+
                                  | For each chunk in Mahabharat...   |
                                  | 1. Generate a concise summary    |
                                  | 2. Prepend: "Summary: ..."       |
                                  | 3. Create Voyage AI embedding    |
                                  | 4. Store in-memory               |
                                  +----------------------------------+
                                                 ^
                                                 | (One-time setup)
                                                 |
[User Query] ----------------> [Voyage AI Embedding]
                                       |
                                       v
          +-------------------------------------------------------+
          | In-Memory Vector Search (Numpy Dot Product)           |
          | against pre-computed, summary-enriched chunk embeddings |
          +-------------------------------------------------------+
                                       |
                                       v
[Top-K Retrieved Contexts] -> [Gemini 2.5 Flash for Generation] -> [Final Answer]
```

### Deployment

The system was split into two components for deployment:

*   **Backend:** A Python/Flask API serving the RAG pipeline, deployed on **Render**.
*   **Frontend:** A clean, simple chat interface built with Next.js, deployed on **Vercel**.

## Future Work & Product-Level Enhancements

While the current system is robust and effective, there are several avenues for future improvement, moving from a proof-of-concept to a product-grade application:

*   **Optimizing Chunk Size:** Experimenting with larger chunk sizes (e.g., 2048 characters) could be beneficial. While the current size is effective, larger chunks might capture more sprawling dialogues or complex descriptions, potentially improving retrieval for broader questions at the cost of precision.
*   **Cross-Book Multi-Hop Evaluation:** The current evaluation set is chapter-based. A truly advanced system would need to answer questions that require synthesizing information from *multiple, disparate chapters* (e.g., "Trace the evolution of Duryodhana's character from the beginning of the book to the start of the war"). Creating such a dataset would be a significant undertaking but would allow for testing more sophisticated retrieval and generation strategies.
*   **Implementing a User Feedback Loop:** The frontend could be enhanced with a simple "thumbs-up/thumbs-down" or a short feedback form for each answer. This real-world data is invaluable for identifying blind spots, understanding user intent, and continuously fine-tuning the model and retrieval process.
*   **LLM Observability and Tracing:** Integrating tools like LangSmith or Arize AI would enable robust monitoring. This allows for tracing the full lifecycle of a query—from the initial question to the retrieved chunks and the final generated answer—making it easier to debug failures, evaluate performance over time, and prevent regressions.

## Conclusion & Connect

This project was a fantastic journey in building a RAG system from the ground up. It reinforced several key lessons:
*   **Intelligent data preparation is paramount.** The context-aware chunking and summary-enrichment steps provided the biggest performance gains.
*   **Don't trust, verify.** Blindly trusting standard tools like RAGAs can be misleading. A custom, well-designed evaluation set is your best friend.
*   **Simplicity can be powerful.** The final, high-performing architecture was simpler than the initial "standard" attempt.

Cheers, and do connect with me for the code or any other nitty-gritties.
