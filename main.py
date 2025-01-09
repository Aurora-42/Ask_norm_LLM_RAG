import chromadb
import cohere
import os

# Configure the key with the environment variable COHERE_TOKEN
COHERE_API_KEY = os.environ.get('COHERE_TOKEN')
cohere_client = cohere.Client(COHERE_API_KEY)

# Path to the ChromaDB database
chrmadb_path = "./chromadb.db"
chroma_client = chromadb.PersistentClient(path=chrmadb_path)
collection = chroma_client.get_collection("my_collection")

prompt_template = "The following documents are relevant to your question:\n{}\n\n"

# Transform the text into a vector using the Cohere embedding model.
def get_embedding(text):
	response = cohere_client.embed(
		texts=[text],
		model="multilingual-22-12"
	)
	return response.embeddings[0]

# Search relevant documents in ChromaDB based on the question.
def search_document(question):
	query_embedding = get_embedding(question)
	results = collection.query(
		query_embeddings=[query_embedding],
		n_results=3
	)
	return results

# Format a list of relevant documents into a string for display.
def format_search_result(relevant_documents):
	formatted_list = []

	for i, doc in enumerate(relevant_documents["documents"][0]):
		formatted_list.append("[{}]: {}".format(relevant_documents["metadatas"][0][i]["source"], doc))

	documents_str = "\n".join(formatted_list)
	return documents_str

# Generate a response to the question using the Cohere Generate model.
def execute_llm(prompt, question):
	response = cohere_client.generate(
		model="command-xlarge-nightly",
		prompt=f"{prompt}\nUser question: {question}\nAnswer:",
		max_tokens=512,
		temperature=0.5
	)
	return response.generations[0].text.strip()

# Main loop of the program.
def run():
	while True:
		question = input("What is your question? ")
		if question.lower() == "exit":
			print("Goodbye!")
			break

		relevant_documents = search_document(question)
		documents_str = format_search_result(relevant_documents)

		prompt = prompt_template.format(documents_str)
		answer = execute_llm(prompt, question)
		print(answer)
		print("\n\n\n")


if __name__ == "__main__":
	run()
