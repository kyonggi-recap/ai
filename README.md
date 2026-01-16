# Fake News Detection System with RAG & AWS SageMaker

This project implements a **Fake News Detection System** utilizing **Retrieval-Augmented Generation (RAG)** architecture. It leverages **AWS SageMaker** for model deployment and inference, and **AWS OpenSearch** as a vector store to retrieve relevant context for accurate classification.

Additionally, the project includes NLP experiments using BART models for text summarization.

## 🏗 System Architecture

The system follows a RAG pipeline to classify news as "Real" or "Fake" by comparing the query with similar contexts.

1.  **Data Processing**: News data is embedded into 1536-dimensional vectors (compatible with OpenAI embeddings).
2.  **Vector Store**: Embeddings are indexed in **AWS OpenSearch Service** using k-NN (k-Nearest Neighbors) with HNSW algorithm.
3.  **Retrieval**: For a given input query, relevant contexts are retrieved from OpenSearch.
4.  **Inference**:
    * The Query + Retrieved Contexts (Context 1, 2, 3) are fed into a model deployed on **AWS SageMaker**.
    * The model predicts the probability of the news being real or fake.

## 🛠 Tech Stack

### Cloud & Infrastructure
* **AWS SageMaker**: Model deployment, endpoint management, and inference.
* **AWS OpenSearch Service**: Vector database for storing and retrieving embeddings.
* **AWS IAM**: Role and permission management.

### AI & Data
* **Python**: Core programming language.
* **PyTorch / Hugging Face Transformers**: Model handling (BART, etc.).
* **Boto3**: AWS SDK for Python to interact with SageMaker and OpenSearch.
* **Pandas / NumPy**: Data manipulation and preprocessing.

## 📂 Project Structure

```text
├── ai_sagemaker_code.ipynb      # Hugging Face BART model deployment on SageMaker
├── aws_sagemaker_code.ipynb     # Summarization & Embedding similarity experiments
├── fake_news_model_deploy.ipynb # Deploying the main Fake News Detection model to SageMaker
├── fake_or_true.ipynb           # Inference script: Invoking the SageMaker endpoint with RAG inputs
├── opensearch_setup.ipynb       # OpenSearch index creation & Data upload (Bulk insert)
├── opensearch_schema.rtf        # JSON Schema for OpenSearch Index (k-NN settings)
├── opensearch_accessrole.rtf    # IAM Policy definitions for OpenSearch access
└── README.md                    # Project Documentation
