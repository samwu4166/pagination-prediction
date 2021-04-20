# import motor.motor_asyncio
from pymongo import MongoClient
from bson.objectid import ObjectId
from settings import MONGO_USER, MONGO_PASSWORD, MONGO_PORT, DATABASE, COLLECTION

# print(f"Loading settings, user:{MONGO_USER}, passwd:{MONGO_PASSWORD}")

MONGO_DETAILS = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@localhost:{MONGO_PORT}"

client = MongoClient(MONGO_DETAILS)

database = client[DATABASE]

page_collection = database.get_collection(COLLECTION)


# helpers
def page_helper(page) -> dict:
    return {
        "tid": str(page["tid"]),
        "url": page["url"],
        "urls": page["urls"],
        "created_time": page["created_time"],
    }


# Retrieve all pages present in the database
def retrieve_pages():
    pages = []
    for page in page_collection.find():
        pages.append(page_helper(page))
    return pages


# Add a new page into to the database
def add_page(page_data: dict) -> dict:
    status_data = {"status": False, "tid": None}
    page = page_collection.insert_one(page_data)
    new_page = page_collection.find_one({"tid": ObjectId(page_data['tid'])})
    status_data["status"] = True
    status_data["tid"] = str(new_page["tid"])
    return status_data


# Retrieve a page with a matching ID
def retrieve_page(tid: str) -> dict:
    page = page_collection.find_one({"tid": ObjectId(tid)})
    if page:
        return page_helper(page)


# Update a page with a matching ID
def update_page(tid: str, data: dict):
    # Return false if an empty request body is sent.
    if len(data) < 1:
        return False
    page = page_collection.find_one({"tid": ObjectId(tid)})
    if page:
        updated_page = page_collection.update_one(
            {"tid": ObjectId(tid)}, {"$set": data}
        )
        if updated_page:
            return True
        return False


# Delete a page from the database
def delete_page(tid: str):
    page = page_collection.find_one({"tid": ObjectId(tid)})
    if page:
        page_collection.delete_one({"tid": ObjectId(tid)})
        return True
