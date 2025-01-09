import chromadb
import cohere
import PyPDF2
import uuid
import os

# Configure API Cohere
COHERE_API_KEY = os.environ.get('COHERE_TOKEN')
cohere_client = cohere.Client(COHERE_API_KEY)

# Path to the ChromaDB database
chrmadb_path = "./chromadb.db"
chroma_client = chromadb.PersistentClient(path=chrmadb_path)

# Veifying if the collection exists
existing_collections = chroma_client.list_collections()
if "my_collection" in existing_collections:
	collection = chroma_client.get_collection("my_collection")
else:
	collection = chroma_client.create_collection(name="my_collection")


# Size of the chunk of text to send to Cohere
CHUNK_SIZE = 1000
# Adjustment to the chunk size to make sure we don't cut off words
OFFSET = 100

# Reading the PDF document and returning text in string format
def get_document(document_path):
	try:
		with open(document_path, 'rb') as file:
			reader = PyPDF2.PdfReader(file)
			document_text = ""
			for page in reader.pages:
				document_text += page.extract_text()
		return document_text
	except Exception as e:
		print(f"Erro ao ler o documento {document_path}: {e}")
		return ""

# Devide the document in a list of strings
def split_document(document_text):
	documents = []
	for i in range(0, len(document_text), CHUNK_SIZE):
		start = i
		end = i + CHUNK_SIZE
		if start != 0:
			start -= OFFSET
			end -= OFFSET
		documents.append(document_text[start:end])
	return documents

# Transform the text into a vector using the Cohere embedding model.
def get_embedding(text):
	try:
		response = cohere_client.embed(
			texts=[text],
			model="multilingual-22-12"
		)
		return response.embeddings[0]
	except Exception as e:
		print(f"Erro ao gerar embedding: {e}")
		return []

# Prepare documents for the database, generating embeddings and metadata
def prepare_documents(documents, document_name):
	embeddings = []
	metadatas = []
	for i, doc in enumerate(documents):
		embedding = get_embedding(doc)
		if embedding:
			embeddings.append(embedding)
			metadatas.append({"source": document_name, "partition": i})
	return embeddings, metadatas

# Create a list of IDs for the documents
def create_ids(documents):
	return [str(uuid.uuid4()) for _ in documents]

# Insert data into the database collection
def insert_data(documents, embeddings, metadatas, ids):
	try:
		collection.add(
			embeddings=embeddings,
			documents=documents,
			metadatas=metadatas,
			ids=ids
		)
		print(f"Dados inseridos com sucesso! {len(documents)} blocos processados.")
	except Exception as e:
		print(f"Erro ao inserir dados na coleção: {e}")

# Execute the processing of the documents and insert into the database
def run():
	print("Preparando os documentos...")
	data_path = './data'

	if not os.path.exists(data_path):
		print(f"Erro: O diretório {data_path} não existe.")
		return

	document_names = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
	if not document_names:
		print(f"Erro: Nenhum arquivo PDF encontrado no diretório {data_path}.")
		return

	documents = []
	embeddings = []
	metadatas = []

	total_documents = len(document_names)
	for i, document_name in enumerate(document_names):
		print(f"{i + 1}/{total_documents}: Processando {document_name}")

		document_path = os.path.join(data_path, document_name)
		document_text = get_document(document_path)
		if not document_text:
			continue

		document_chunks = split_document(document_text)
		document_embeddings, document_metadatas = prepare_documents(document_chunks, document_name)

		documents.extend(document_chunks)
		embeddings.extend(document_embeddings)
		metadatas.extend(document_metadatas)

	if documents:
		ids = create_ids(documents)
		insert_data(documents, embeddings, metadatas, ids)
	else:
		print("Nenhum documento foi processado.")

if __name__ == "__main__":
	run()
