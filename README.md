# Document Processing with LLM + RAG

This project process documents, extracts text from PDF's, and generates embeddings using Cohere's multilingual model. The processed data is stored in a ChromaDB database, which can later be used for retrieval-augmented generation (RAG) with language models (LLMs).

In this case I am using the document The Norm from 42.

## Prerequisites

Before starting, ensure you have the following installed:
* Docker;
* Docker Compose;
* <a href="https://cohere.com/" >A Cohere API key </a>;

## Setup and Usage

1. Clone the repository

	```
	git clone git@github.com:Aurora-42/Ask_norm_LLM_RAG.git

	cd Ask_norm_LLM_RAG

	```

2. Environment Variables

	```
	export COHERE_TOKEN="your_token"
	```

3. Build and Run with Docker

	Build the docker image:
	```
	docker-compose up --build -d
	```

	Run the container:
	```
	docker exec -it ask_norm_LLM_RAG-openai-app-1 bash
	```

4. Process the document

	By running the command bellow, the processed embeddings and metadata will be stored in the ChromaDB database at ./chromadb.db.

	```
	python prep_docs.py
	```
5. Chatting with AI about the document

	```
	python main.py
	```
