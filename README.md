# Car Insurance RAG Application

A Retrieval-Augmented Generation (RAG) application designed to answer questions about car insurance based on a provided FAQ document.

![Application Demo](https://github.com/user-attachments/assets/531b98c9-ba5a-48cc-8235-6a2f48e6bfca)

## Overview

This project implements a QA system that uses semantic search to retrieve relevant information from a car insurance FAQ and then generates a detailed answer using a Large Language Model (LLM).

## Key Features

-   **Semantic Search**: Utilizes `sentence-transformers/all-MiniLM-L6-v2` to create vector embeddings of the FAQ content.
-   **Efficient Retrieval**: Uses **FAISS** (Facebook AI Similarity Search) for fast and accurate similarity searches.
-   **LLM Integration**: Integrates with the **Vextapp API** (using **Gemini 2.0 Flash**) for generating human-like responses based on the retrieved context.
-   **Local Caching**: Embeddings and the FAISS index are cached locally (`paragraph_embeddings.npy` and `faiss_index.bin`) to speed up subsequent runs.
-   **Interactive CLI**: Simple command-line interface for user queries.

## Prerequisites

-   Python 3.7+
-   A Vextapp API Key and Channel Token.

## Installation

1.  Clone the repository.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The application requires two environment variables to interact with the Vextapp API:

-   `VEXTAPP_CHANNEL_TOKEN`: Your Vextapp channel token.
-   `VEXTAPP_API_KEY`: Your Vextapp API key.

You can set these in your terminal:

```bash
export VEXTAPP_CHANNEL_TOKEN='your_channel_token'
export VEXTAPP_API_KEY='your_api_key'
```

## Usage

Run the application using:

```bash
python rag_car_insurance.py
```

Once the system is ready, you can ask questions like:
-   "What is third-party insurance?"
-   "How are premiums calculated?"
-   "What is a No Claim Bonus (NCB)?"

Type `exit` to quit the application.

## Project Structure

-   `rag_car_insurance.py`: Main application script.
-   `car_insurance_faq.txt`: Source document containing FAQ paragraphs.
-   `requirements.txt`: Python dependencies.
-   `faiss_index.bin`: (Generated) FAISS index for vector search.
-   `paragraph_embeddings.npy`: (Generated) Cached embeddings of the FAQ paragraphs.

## Contact

If you have any queries, feedback, or suggestions, feel free to drop a mail at chetan.mahale0220@gmail.com :)
