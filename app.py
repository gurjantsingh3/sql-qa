from llama_index.core.query_engine import SQLAutoVectorQueryEngine, RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
import openai
import os
from flask import Flask, render_template, request, jsonify
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
import sqlalchemy
from llama_index.core import SQLDatabase
from dotenv import load_dotenv
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core import Settings
from sqlalchemy.sql import select
from llama_index.llms.openai import OpenAI
import chromadb
from llama_index.core import Document
openai.api_key = os.environ.get("OPENAI_API_KEY")
load_dotenv()

app = Flask(__name__)
collection_name = "sql-db"
chroma_client = chromadb.PersistentClient()
llm = OpenAI(model="gpt-4")
# llm = Ollama(model="llama2")
# resp = llm.complete("Who is Paul Graham?")

# print(resp)

try:
    # Try to get the existing collection
    chroma_collection = chroma_client.get_collection(collection_name)
    # print("Collection '{}' already exists.".format(collection_name))
except:
    # If collection does not exist, create it
    chroma_collection = chroma_client.create_collection(collection_name)
    # print("Collection '{}' created.".format(collection_name))

# Create ChromaVectorStore using the existing or newly created collection
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex([], storage_context=storage_context)
print(vector_index, "vector_index")


@app.route('/')
def index():
    return render_template('index.html')


DatabaseUrl = os.getenv("DATABASE_URL")
engine = sqlalchemy.create_engine(DatabaseUrl)
sql_database = SQLDatabase(engine)
sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database
)


# @app.route('/vector_indexing', methods=['POST'])
def vector_indexing():
    # Get metadata about all tables in the database
    metadata = sqlalchemy.MetaData()
    metadata.reflect(bind=engine)
    print(metadata)
    conn = engine.connect()
    # Iterate over each table

    all_documents = []

    # Iterate over each table in the metadata
    for table_name in metadata.tables.keys():
        table = metadata.tables[table_name]

        # Get columns of the table and extract column names
        column_names = [column.name for column in table.columns]

        # Construct a document containing the table name, column names, and row data
        document_text = f"Table: {table_name},\n Columns: {', '.join(column_names)}\n"

        # Construct select statement with all columns
        query = select(table)

        # Execute the query and process the results
        result = conn.execute(query)

        # Iterate over the results and append row data to the document
        for row in result:
            row_data = ', '.join(str(column) for column in row)
            document_text += f"Row data: {row_data}\n"

        # Create a document object
        document = Document(id_=table_name, text=document_text)

        # Append the document to the list
        all_documents.append(document)

    # Insert all documents into the vector index
    nodes = Settings.node_parser.get_nodes_from_documents(all_documents)
    vector_index.insert_nodes(nodes)
    return jsonify({"success": True})


# vector_indexing()


@app.route("/query", methods=["POST"])
def query_engine():
    query = request.json["question"]
    vector_store_info = VectorStoreInfo(
        content_info="Tables data are used to store data in the database",
        metadata_info=[
            MetadataInfo(
                name="tables",
                type="string",
                description="Tables name in the database",
            ),
        ],
    )

    vector_auto_retriever = VectorIndexAutoRetriever(
        vector_index, vector_store_info=vector_store_info
    )

    retriever_query_engine = RetrieverQueryEngine.from_args(
        vector_auto_retriever, llm=llm
    )

    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        description=(
            "Useful for executing any SQL query on the database. "
            "This tool does not allow insert, delete, or alter actions."
            "Input should be a SQL query."
        ),
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=retriever_query_engine,
        description=(
            "Useful for retrieving data from the database. "
            "This tool does not allow insert, delete, or alter actions."
            "Input should be a string. "
        )

    )
    query_engine = SQLAutoVectorQueryEngine(
        sql_tool, vector_tool, llm=llm
    )
    response = query_engine.query(query)
    response_str = str(response)
    return jsonify({"output": response_str}), 200


if __name__ == '__main__':
    app.run(debug=False, port=8000, host="0.0.0.0")
