from pika_queue import PikaQueue
from settings import RABBIT_ACCOUNT, RABBIT_PASSWORD, RABBIT_HOST, RABBIT_PORT, EVENT_QUEUE, TEST_WORKER_QUEUE
import os
import sys
from bson.objectid import ObjectId
from datetime import datetime
#import page CRUDs
from database import (
    add_page
)
from publisher import publish_message, publish_error
import validators
sys.path.insert(0, os.path.abspath(r".."))
from autopager.autopager import get_shared_autopager
from autopager.preprocessing import generate_page_component

QUEUE_NAME = EVENT_QUEUE
working_queue = PikaQueue(RABBIT_HOST, RABBIT_PORT, RABBIT_ACCOUNT, RABBIT_PASSWORD)

def main():
    #working_queue.DeclareQueue(QUEUE_NAME)
    def callback(ch, method, properties, body):
        decoded_url = body.decode()
        decoded_url = decoded_url.strip('"')
        print(" [x] Received %r" % decoded_url)
        try:
            if validators.url(decoded_url):
                page_workflow(decoded_url)
            else:
                print("Not a valid url, pass task.")
        except Exception as e:
            if QUEUE_NAME == EVENT_QUEUE:
                publish_error(decoded_url)
            print(f"Error! url: {decoded_url} ({e})")
        print(" [x] Done")
        ch.basic_ack(delivery_tag=method.delivery_tag)
    def page_workflow(page_url):
        _uid = ObjectId()
        uncoded_url = page_url
        # print(f"Get request: {uncoded_url}")
        autopager = get_shared_autopager()
        page_component = generate_page_component(uncoded_url)
        result_urls = autopager.urls(page_component["html"], uncoded_url, direct=True, prev=False, next=False)
        current_time = datetime.now()
        result_component = {"tid": _uid, "url": uncoded_url, "urls": result_urls, "created_time": current_time}
        ### add result to Mongo
        new_page = add_page(result_component)
        if new_page["status"] == False or new_page["tid"] == None:
            print("Bypass this url. (Document already exist.)")
        else:
            if QUEUE_NAME == EVENT_QUEUE:
                print("Publish to ExtractQueue.")
                publish_message(new_page["tid"])
            print("Success! result tid: ",new_page["tid"])
    working_queue.GetFromQueue(QUEUE_NAME, callback, False)
def nothing(ch, method, properties, body):
    print(body)
if __name__ == "__main__":
    try:
        print("Start consuming data")
        main()
    except KeyboardInterrupt:
        print("Interrupted, stop consuming data...")
        working_queue.Close()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    #print(working_queue.GetFromQueue(QUEUE_NAME, nothing, False))