import asyncio
import os
import sys
from datetime import datetime
import logging

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# Mock the config to avoid import issues
class MockConfig:
    @staticmethod
    def get(key, default=None):
        config_map = {
            "vector_dimensions": "3072",
            "openai_host": os.getenv("TEST_MONGODB_HOST", "mongodb://user:pass@localhost:27017/?directConnection=true"),
            "openai_db_name": "openai_db",
            "vector_collection_name": "test_vectors",
            "vector_index_name": "test_vector_index",
        }
        return config_map.get(key, default)

# Mock the modules before importing
sys.modules['open_webui.env'] = type('MockEnv', (), {'SRC_LOG_LEVELS': {}})
sys.modules['open_webui.config'] = type('MockConfig', (), {'MONGODB_CONFIG': MockConfig()})

# Now import the mongo_db module
from mongo_db import (
    mongodb_client,
    get_mongodb_session,
    connect,
    query,
    query_one,
    insert,
    update,
    delete_one,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Test configuration
TEST_DB_NAME = "openai_db"
TEST_COLLECTION_NAME = "test_collection"
MONGODB_HOST = os.getenv("TEST_MONGODB_HOST", "mongodb://user:pass@localhost:27017/?directConnection=true")

class MongoDBFunctionTester:
    """Test all functions from mongo_db.py"""
    
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
    
    def log_test_result(self, test_name: str, success: bool, message: str = ""):
        """Log test results."""
        status = "✅ PASS" if success else "❌ FAIL"
        full_message = f"{status} - {test_name}"
        if message:
            full_message += f": {message}"
        
        logger.info(full_message)
        self.test_results.append((test_name, success, message))
        
        if not success:
            self.failed_tests.append(test_name)
    
    async def test_connection_functions(self):
        """Test connection-related functions."""
        logger.info("\n🔌 Testing Connection Functions")
        
        try:
            # Test global connect function
            await connect()
            self.log_test_result("connect()", True, "Global connect function works")
            
            # Test if mongodb_client is connected
            if mongodb_client._connected:
                self.log_test_result("mongodb_client._connected", True, "Client connected successfully")
            else:
                self.log_test_result("mongodb_client._connected", False, "Client not connected")
                return False
            
            # Test get_client methods
            try:
                sync_client = await mongodb_client.get_client()
                self.log_test_result("get_client()", True, "Sync client retrieved")
            except Exception as e:
                self.log_test_result("get_client()", False, str(e))
            
            try:
                async_client = await mongodb_client.get_async_client()
                self.log_test_result("get_async_client()", True, "Async client retrieved")
            except Exception as e:
                self.log_test_result("get_async_client()", False, str(e))
            
            return True
            
        except Exception as e:
            self.log_test_result("Connection Functions", False, str(e))
            return False
    
    async def test_insert_functions(self):
        """Test insert functions."""
        logger.info("\n📝 Testing Insert Functions")
        
        try:
            # Test global insert function
            test_doc = {
                "name": "test_user_global",
                "email": "global@example.com",
                "age": 30,
                "created_at": datetime.utcnow()
            }
            
            result = await insert(TEST_DB_NAME, TEST_COLLECTION_NAME, test_doc)
            
            if result and "_id" in result:
                self.log_test_result("insert() - global function", True, f"Document inserted with ID: {result['_id']}")
            else:
                self.log_test_result("insert() - global function", False, "No result returned")
                return False
            
            # Test MongoDBClient.insert method
            test_doc2 = {
                "name": "test_user_client",
                "email": "client@example.com",
                "age": 25,
                "created_at": datetime.utcnow()
            }
            
            result2 = await mongodb_client.insert(TEST_DB_NAME, TEST_COLLECTION_NAME, test_doc2)
            
            if result2 and "_id" in result2:
                self.log_test_result("MongoDBClient.insert()", True, f"Document inserted with ID: {result2['_id']}")
            else:
                self.log_test_result("MongoDBClient.insert()", False, "No result returned")
            
            # Test insert_many method
            test_docs = [
                {"name": "bulk_user1", "department": "engineering", "salary": 80000},
                {"name": "bulk_user2", "department": "marketing", "salary": 60000},
                {"name": "bulk_user3", "department": "engineering", "salary": 90000}
            ]
            
            bulk_result = await mongodb_client.insert_many(TEST_DB_NAME, TEST_COLLECTION_NAME, test_docs)
            
            if bulk_result.get("success") and bulk_result.get("inserted_count") == 3:
                self.log_test_result("MongoDBClient.insert_many()", True, f"Inserted {bulk_result['inserted_count']} documents")
            else:
                self.log_test_result("MongoDBClient.insert_many()", False, "Bulk insert failed")
            
            return True
            
        except Exception as e:
            self.log_test_result("Insert Functions", False, str(e))
            return False
    
    async def test_query_functions(self):
        """Test query functions."""
        logger.info("\n🔍 Testing Query Functions")
        
        try:
            # Test global query function
            results = await query(TEST_DB_NAME, TEST_COLLECTION_NAME)
            
            if isinstance(results, list) and len(results) > 0:
                self.log_test_result("query() - global function", True, f"Retrieved {len(results)} documents")
            else:
                self.log_test_result("query() - global function", False, "No documents retrieved")
            
            # Test global query_one function
            single_result = await query_one(TEST_DB_NAME, TEST_COLLECTION_NAME, {"name": "test_user_global"})
            
            if single_result and single_result.get("name") == "test_user_global":
                self.log_test_result("query_one() - global function", True, "Found specific document")
            else:
                self.log_test_result("query_one() - global function", False, "Document not found")
            
            # Test MongoDBClient.query method
            client_results = await mongodb_client.query(TEST_DB_NAME, TEST_COLLECTION_NAME)
            
            if isinstance(client_results, list) and len(client_results) > 0:
                self.log_test_result("MongoDBClient.query()", True, f"Retrieved {len(client_results)} documents")
            else:
                self.log_test_result("MongoDBClient.query()", False, "No documents retrieved")
            
            # Test MongoDBClient.query_one method
            client_single = await mongodb_client.query_one(TEST_DB_NAME, TEST_COLLECTION_NAME, {"name": "test_user_client"})
            
            if client_single and client_single.get("name") == "test_user_client":
                self.log_test_result("MongoDBClient.query_one()", True, "Found specific document")
            else:
                self.log_test_result("MongoDBClient.query_one()", False, "Document not found")
            
            # Test query with filter
            filtered_results = await mongodb_client.query(TEST_DB_NAME, TEST_COLLECTION_NAME, {"department": "engineering"})
            
            if isinstance(filtered_results, list) and len(filtered_results) >= 2:
                self.log_test_result("query() with filter", True, f"Found {len(filtered_results)} engineering documents")
            else:
                self.log_test_result("query() with filter", False, "Filtered query failed")
            
            # Test query_one_with_projection
            projected_result = await mongodb_client.query_one_with_projection(
                TEST_DB_NAME, 
                TEST_COLLECTION_NAME, 
                {"name": "test_user_global"}, 
                {"name": 1, "email": 1, "_id": 0}
            )
            
            if projected_result and "name" in projected_result and "email" in projected_result and "_id" not in projected_result:
                self.log_test_result("query_one_with_projection()", True, "Projection worked correctly")
            else:
                self.log_test_result("query_one_with_projection()", False, "Projection failed")
            
            # Test query_with_sort_and_pagination
            paginated_result = await mongodb_client.query_with_sort_and_pagination(
                TEST_DB_NAME,
                TEST_COLLECTION_NAME,
                {},
                sort_field="name",
                sort_order=1,
                limit=3,
                skip=0
            )
            
            if (paginated_result.get("count") is not None and 
                paginated_result.get("total") is not None and
                isinstance(paginated_result.get("documents"), list)):
                self.log_test_result("query_with_sort_and_pagination()", True, 
                                   f"Retrieved {paginated_result['count']} of {paginated_result['total']} documents")
            else:
                self.log_test_result("query_with_sort_and_pagination()", False, "Pagination failed")
            
            return True
            
        except Exception as e:
            self.log_test_result("Query Functions", False, str(e))
            return False
    
    async def test_update_functions(self):
        """Test update functions."""
        logger.info("\n✏️ Testing Update Functions")
        
        try:
            # Test global update function
            update_result = await update(
                TEST_DB_NAME, 
                TEST_COLLECTION_NAME, 
                {"name": "test_user_global"}, 
                {"age": 31, "updated": True}
            )
            
            if update_result and update_result.get("age") == 31:
                self.log_test_result("update() - global function", True, "Document updated successfully")
            else:
                self.log_test_result("update() - global function", False, "Update failed")
            
            # Test MongoDBClient.update method
            client_update_result = await mongodb_client.update(
                TEST_DB_NAME,
                TEST_COLLECTION_NAME,
                {"name": "test_user_client"},
                {"age": 26, "updated": True}
            )
            
            if client_update_result and client_update_result.get("age") == 26:
                self.log_test_result("MongoDBClient.update()", True, "Document updated successfully")
            else:
                self.log_test_result("MongoDBClient.update()", False, "Update failed")
            
            # Test update_many method
            many_update_result = await mongodb_client.update_many(
                TEST_DB_NAME,
                TEST_COLLECTION_NAME,
                {"department": "engineering"},
                {"bonus": 5000}
            )
            
            if many_update_result.get("matched_count", 0) >= 2:
                self.log_test_result("MongoDBClient.update_many()", True, 
                                   f"Updated {many_update_result['modified_count']} documents")
            else:
                self.log_test_result("MongoDBClient.update_many()", False, "Bulk update failed")
            
            return True
            
        except Exception as e:
            self.log_test_result("Update Functions", False, str(e))
            return False
    
    async def test_aggregation_functions(self):
        """Test aggregation functions."""
        logger.info("\n📊 Testing Aggregation Functions")
        
        try:
            # Test query_one_by_pipeline
            pipeline = [
                {
                    "$group": {
                        "_id": "$department",
                        "avg_salary": {"$avg": "$salary"},
                        "count": {"$sum": 1}
                    }
                },
                {"$sort": {"avg_salary": -1}},
                {"$limit": 1}
            ]
            
            agg_result = await mongodb_client.query_one_by_pipeline(
                TEST_DB_NAME,
                TEST_COLLECTION_NAME,
                pipeline
            )
            
            if agg_result and "_id" in agg_result and "avg_salary" in agg_result:
                self.log_test_result("query_one_by_pipeline()", True, 
                                   f"Aggregation returned: {agg_result['_id']} with avg salary {agg_result['avg_salary']}")
            else:
                self.log_test_result("query_one_by_pipeline()", False, "Aggregation failed")
            
            return True
            
        except Exception as e:
            self.log_test_result("Aggregation Functions", False, str(e))
            return False
    
    async def test_delete_functions(self):
        """Test delete functions."""
        logger.info("\n🗑️ Testing Delete Functions")
        
        try:
            # Test global delete_one function
            delete_result = await delete_one(TEST_DB_NAME, TEST_COLLECTION_NAME, {"name": "test_user_global"})
            
            if delete_result.get("deleted_count", 0) == 1:
                self.log_test_result("delete_one() - global function", True, "Document deleted successfully")
            else:
                self.log_test_result("delete_one() - global function", False, "Delete failed")
            
            # Test MongoDBClient.delete_one method
            client_delete_result = await mongodb_client.delete_one(TEST_DB_NAME, TEST_COLLECTION_NAME, {"name": "test_user_client"})
            
            if client_delete_result.get("deleted_count", 0) == 1:
                self.log_test_result("MongoDBClient.delete_one()", True, "Document deleted successfully")
            else:
                self.log_test_result("MongoDBClient.delete_one()", False, "Delete failed")
            
            # Test delete_many method
            many_delete_result = await mongodb_client.delete_many(TEST_DB_NAME, TEST_COLLECTION_NAME, {"department": "engineering"})
            
            if many_delete_result.get("deleted_count", 0) >= 2:
                self.log_test_result("MongoDBClient.delete_many()", True, 
                                   f"Deleted {many_delete_result['deleted_count']} documents")
            else:
                self.log_test_result("MongoDBClient.delete_many()", False, "Bulk delete failed")
            
            return True
            
        except Exception as e:
            self.log_test_result("Delete Functions", False, str(e))
            return False
    
    async def test_context_manager(self):
        """Test context manager function."""
        logger.info("\n🔄 Testing Context Manager")
        
        try:
            async with get_mongodb_session() as mongo:
                # Test that we get the mongodb_client instance
                if mongo == mongodb_client:
                    self.log_test_result("get_mongodb_session() - returns client", True, "Context manager returns correct client")
                else:
                    self.log_test_result("get_mongodb_session() - returns client", False, "Context manager returns wrong object")
                
                # Test using the context manager for operations
                test_doc = {"context_test": True, "value": 42}
                result = await mongo.insert(TEST_DB_NAME, TEST_COLLECTION_NAME, test_doc)
                
                if result and "_id" in result:
                    self.log_test_result("get_mongodb_session() - operations", True, "Operations work through context manager")
                    
                    # Clean up
                    await mongo.delete_one(TEST_DB_NAME, TEST_COLLECTION_NAME, {"context_test": True})
                else:
                    self.log_test_result("get_mongodb_session() - operations", False, "Operations failed through context manager")
            
            return True
            
        except Exception as e:
            self.log_test_result("Context Manager", False, str(e))
            return False
    
    async def test_error_handling(self):
        """Test error handling in functions."""
        logger.info("\n⚠️ Testing Error Handling")
        
        try:
            # Test query with non-existent collection
            empty_result = await query("nonexistent_db", "nonexistent_collection")
            if isinstance(empty_result, list) and len(empty_result) == 0:
                self.log_test_result("Error Handling - query nonexistent", True, "Returns empty list for nonexistent collection")
            else:
                self.log_test_result("Error Handling - query nonexistent", False, "Doesn't handle nonexistent collection properly")
            
            # Test query_one with non-existent document
            none_result = await query_one(TEST_DB_NAME, TEST_COLLECTION_NAME, {"name": "nonexistent_user"})
            if none_result is None:
                self.log_test_result("Error Handling - query_one nonexistent", True, "Returns None for nonexistent document")
            else:
                self.log_test_result("Error Handling - query_one nonexistent", False, "Doesn't handle nonexistent document properly")
            
            # Test update with non-existent document
            update_result = await update(TEST_DB_NAME, TEST_COLLECTION_NAME, {"name": "nonexistent"}, {"value": 1})
            if update_result is None:
                self.log_test_result("Error Handling - update nonexistent", True, "Returns None for nonexistent document update")
            else:
                self.log_test_result("Error Handling - update nonexistent", False, "Doesn't handle nonexistent document update properly")
            
            # Test delete with non-existent document
            delete_result = await delete_one(TEST_DB_NAME, TEST_COLLECTION_NAME, {"name": "nonexistent"})
            if delete_result.get("deleted_count", 0) == 0:
                self.log_test_result("Error Handling - delete nonexistent", True, "Returns 0 deleted_count for nonexistent document")
            else:
                self.log_test_result("Error Handling - delete nonexistent", False, "Doesn't handle nonexistent document delete properly")
            
            return True
            
        except Exception as e:
            self.log_test_result("Error Handling", False, str(e))
            return False
    
    async def cleanup_test_data(self):
        """Clean up all test data."""
        try:
            await mongodb_client.delete_many(TEST_DB_NAME, TEST_COLLECTION_NAME, {})
            logger.info("🧹 Test data cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup error: {e}")
    
    async def run_all_tests(self):
        """Run all tests and return summary."""
        logger.info("🚀 Starting MongoDB Function Integration Tests")
        logger.info(f"📍 MongoDB Host: {MONGODB_HOST}")
        logger.info(f"📍 Test Database: {TEST_DB_NAME}")
        logger.info("=" * 80)
        
        test_functions = [
            self.test_connection_functions,
            self.test_insert_functions,
            self.test_query_functions,
            self.test_update_functions,
            self.test_aggregation_functions,
            self.test_delete_functions,
            self.test_context_manager,
            self.test_error_handling,
        ]
        
        passed_categories = 0
        
        for test_func in test_functions:
            try:
                if await test_func():
                    passed_categories += 1
            except Exception as e:
                logger.error(f"❌ Test category {test_func.__name__} failed with exception: {e}")
        
        # Cleanup
        await self.cleanup_test_data()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 TEST SUMMARY")
        logger.info(f"Categories passed: {passed_categories}/{len(test_functions)}")
        
        passed_tests = len([r for r in self.test_results if r[1]])
        total_tests = len(self.test_results)
        logger.info(f"Total tests passed: {passed_tests}/{total_tests}")
        
        if self.failed_tests:
            logger.info(f"❌ Failed tests: {', '.join(self.failed_tests)}")
        
        if passed_categories == len(test_functions) and len(self.failed_tests) == 0:
            logger.info("🎉 All tests passed!")
            return True
        else:
            logger.info("💥 Some tests failed!")
            return False
    
    async def close_connections(self):
        """Close MongoDB connections."""
        try:
            await mongodb_client.close()
            logger.info("🔌 MongoDB connections closed")
        except Exception as e:
            logger.warning(f"⚠️ Error closing connections: {e}")

async def main():
    """Main test runner."""
    tester = MongoDBFunctionTester()
    
    try:
        success = await tester.run_all_tests()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        await tester.close_connections()

if __name__ == "__main__":
    # Set environment variables for testing
    os.environ["ENV"] = "test"
    
    # Run the tests
    exit_code = asyncio.run(main())
    exit(exit_code)