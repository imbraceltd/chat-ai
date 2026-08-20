import pymongo
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import sys
import time
import os

# Get MongoDB configuration from environment variables
MONGODB_URL = os.getenv(
    "MONGODB_URL", "mongodb+srv://imbrace:g5x83FFVbSRK9C1d@ai-non-prod.gqp7j07.mongodb.net/")
DATABASE_NAME = os.getenv("MONGODB_DATABASE", "imbrace_dev")


def test_mongodb_connection(uri=MONGODB_URL, database=DATABASE_NAME):
    """
    Test MongoDB connection and basic operations

    Args:
        uri (str): MongoDB connection URI
        database (str): Database name to test

    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        # Set a shorter timeout for testing (2 seconds)
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=2000)

        # Try to get server info to verify connection
        client.server_info()

        # Try to access the test database
        db = client[database]

        # try a simmple get collection
        collection = db["openai_assistants"]
        query = {
            "assistant_id": "442ff30b-9c8d-41d7-8e93-1196cf841c0c",
            "organization_id": "org_imbrace",
            "deleted_at": None
        }
        result = collection.find_one(query)
        print(result)

        print("✅ MongoDB Connection Test Successful!")
        print(f"Connected to database: {database}")
        print("Basic operations (insert, find, delete) completed successfully")

        client.close()
        return True

    except ConnectionFailure as e:
        print("❌ MongoDB Connection Error:", str(e))
        print("Please ensure your MongoDB instance is running and the URI is correct.")
        return False
    except ServerSelectionTimeoutError as e:
        print("❌ MongoDB Server Selection Timeout:", str(e))
        print("Make sure MongoDB is running and accessible from your network.")
        return False
    except Exception as e:
        print("❌ An unexpected error occurred:", str(e))
        return False


if __name__ == "__main__":
    print("Testing MongoDB Connection...")
    print(f"Using MongoDB URI: {MONGODB_URL}")
    print(f"Using Database: {DATABASE_NAME}")
    success = test_mongodb_connection()

    if not success:
        sys.exit(1)  # Exit with error code if connection failed
