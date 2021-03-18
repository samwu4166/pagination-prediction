import motor.motor_asyncio
from bson.objectid import ObjectId
from .settings import MONGO_USER, MONGO_PASSWORD

# print(f"Loading settings, user:{MONGO_USER}, passwd:{MONGO_PASSWORD}")

MONGO_DETAILS = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@localhost:27000"

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)

database = client.pageDB

page_collection = database.get_collection("predictResult")


# helpers
def page_helper(page) -> dict:
    return {
        "tid": str(page["tid"]),
        "url": page["url"],
        "urls": page["urls"],
        "created_time": page["created_time"],
    }


# Retrieve all pages present in the database
async def retrieve_pages():
    pages = []
    async for page in page_collection.find():
        pages.append(page_helper(page))
    return pages


# Add a new page into to the database
async def add_page(page_data: dict) -> dict:
    page = await page_collection.insert_one(page_data)
    new_page = await page_collection.find_one({"tid": ObjectId(page_data['tid'])})
    return page_helper(new_page)


# Retrieve a page with a matching ID
async def retrieve_page(tid: str) -> dict:
    page = await page_collection.find_one({"tid": ObjectId(tid)})
    if page:
        return page_helper(page)


# Update a page with a matching ID
async def update_page(tid: str, data: dict):
    # Return false if an empty request body is sent.
    if len(data) < 1:
        return False
    page = await page_collection.find_one({"tid": ObjectId(tid)})
    if page:
        updated_page = await page_collection.update_one(
            {"tid": ObjectId(tid)}, {"$set": data}
        )
        if updated_page:
            return True
        return False


# Delete a page from the database
async def delete_page(tid: str):
    page = await page_collection.find_one({"tid": ObjectId(tid)})
    if page:
        await page_collection.delete_one({"tid": ObjectId(tid)})
        return True