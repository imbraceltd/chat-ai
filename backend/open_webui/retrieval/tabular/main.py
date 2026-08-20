from open_webui.retrieval.tabular.types.tabular import TabularFileProcessor

# Example file IDs
arr = [
   { "id": "e19c9929-5fa3-4535-990e-2cf04f698b22",
    "type": "xlsx",},
    { "id": "f1fea9b6-ad4a-40a7-bd1a-6aee4efbce68",
    "type": "xlsx",},
    { "id": "22116f6a-56ef-4371-ba4e-5c4c53c4af02",
    "type": "xlsx",},
]

# Initialize CSV processor (no longer needs DuckDBClient)
processor = TabularFileProcessor()

# Process each file individually
for file in arr:
    try:
        processor.validateFile()
        print(processor.summarize(file["id"]))
        print(processor.top(file["id"]))
        print(processor.describe(file["id"]))
    except Exception as e:
        print(f"Error processing file {file['id']}: {e}")

# Example of querying multiple files
try:
    results = processor.query(
        [
            "b1d25a8c-3237-4194-98ab-2f38835b38ec",
            "f1fea9b6-ad4a-40a7-bd1a-6aee4efbce68", 
        ],
     
        "SELECT * FROM shop_order_reports_b1d25a8c_3237_4194_98ab_2f38835b38ec UNION ALL SELECT * FROM shop_order_reports_f1fea9b6_ad4a_40a7_bd1a_6aee4efbce68"
     
    )
    print(f"Query results: {results}")
except Exception as e:
    print(f"Query error: {e}")

# try:
#     duckdb_client = DuckDBClient(in_memory=True)
#     data = processor.validateFile()

# except ValueError as e:
#     print(f"Error: {str(e)}")
# except Exception as e:
#     print(f"Unexpected error: {str(e)}")
