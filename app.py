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
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core import Settings
from sqlalchemy.sql import select
from llama_index.llms.openai import OpenAI
import chromadb
from llama_index.core import Document
openai.api_key = os.environ.get("OPENAI_API_KEY")
load_dotenv()

DatabaseUrl = os.getenv("DATABASE_URL")
print(DatabaseUrl)
app = Flask(__name__)
collection_name = "sql-db"
chroma_client = chromadb.PersistentClient()
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

try:
    chroma_collection = chroma_client.get_collection(collection_name)
except:
    chroma_collection = chroma_client.create_collection(collection_name)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex([], storage_context=storage_context)


@app.route('/')
def index():
    return render_template('index.html')


engine = sqlalchemy.create_engine(DatabaseUrl)

selected_tables = ["ActivityLog",
                   "AspNetUsers",
                   "Category",
                   "Checklists",
                   "ChecklistFields",
                   "ChecklistFieldTypes",
                   "ChecklistSections",
                   "ChecklistStatus",
                   "ChecklistSubmissions",
                   "Customer",
                   "DailyWorkEmployees",
                   "Departments",
                   "EmployeeLeaves",
                   "Employees",
                   "Gridowners",
                   "GridStatus",
                   "Incidents",
                   "IncidentStatus",
                   "IncidentFiles",
                   "InstallerActivityLog",
                   "Issues",
                   "IssueStatus",
                   "LeaveTypes",
                   "LeaveStatus",
                   "MessageTemplates",
                   "NonProjectWorkHours",
                   "NonProjectWorkTypes",
                   "OrderChecklists",
                   "OrderComments",
                   "OrderCoordination",
                   "OrderEmployees",
                   "OrderEquipments",
                   "OrderImage",
                   "OrderInverterImages",
                   "OrderInverters",
                   "OrderItem",
                   "OrderLogistics",
                   "OrderMessages",
                   "OrderMeterDetails",
                   "OrderProducts",
                   "Orders",
                   "OrderShipments",
                   "OrdersHistory",
                   "OrderStatus",
                   "OrderTeams",
                   "OrderUsers",
                   "OtherProjects",
                   "OtherTasks",
                   "PanelTypes",
                   "Partner",
                   "PartnerUsers",
                   "Phase",
                   "PriorityProjects",
                   "PriceMatrix",
                   "Product",
                   "Products",
                   "ProjectsListResponse",
                   "QuoteDocuments",
                   "QuoteNotes",
                   "Quotes",
                   "QuotesUsers",
                   "ReferenceIncidents",
                   "Roles",
                   "RoofAreaGeometry",
                   "RoofAreaParts",
                   "RoofAreas",
                   "RoofTypes",
                   "Scheduler",
                   "SchedulerEmployees",
                   "Status",
                   "Teams",
                   "TimeKeeping",
                   "TripletexExportLog",
                   "Users",
                   "UsersIssueAssignment"]
sql_database = SQLDatabase(engine=engine, include_tables=selected_tables)
sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database, verbose=True, llm=llm
)

'''
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
        document_text = f"Table: {table_name},\n Columns: {
            ', '.join(column_names)}\n"

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
    return True

# calling the function
# vector_indexing()
'''


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
            "This tool is designed for querying the database using SQL. "
            "You can ask any question or execute any complex SQL query to retrieve data. "
            "For example, you can use SELECT statements to fetch data, JOIN clauses to combine data from multiple tables, "
            "or WHERE clauses to filter results based on specific conditions."
            "Filter results based on specific conditions."
        ),
    )
    # Setting a detailed description for the Vector tool
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=retriever_query_engine,
        description=(
            "It is particularly useful for answering questions related to document similarity or content retrieval."
            "It can be used to retrieve data from the vector index using the vector_auto_retriever. "
        )


    )
    query_engine = SQLAutoVectorQueryEngine(
        sql_tool, vector_tool, llm=llm, verbose=True
    )

    # Combine Response from SQL tool and Response from Vector tool
    response = query_engine.query(query)

    # Only Response from SQL tool
    # response = sql_query_engine.query(query)
    response_str = str(response)
    return jsonify({"output": response_str}), 200


if __name__ == '__main__':
    app.run(debug=False, port=8000, host="0.0.0.0")
