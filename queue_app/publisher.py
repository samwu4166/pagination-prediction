from pika_queue import PikaQueue
from settings import RABBIT_ACCOUNT, RABBIT_PASSWORD, RABBIT_HOST, RABBIT_PORT, ETL_QUEUE, PREDICT_ERROR_QUEUE

def publish_message(result_tid):
    publish_queue = PikaQueue(RABBIT_HOST, RABBIT_PORT, RABBIT_ACCOUNT, RABBIT_PASSWORD)
    publish_queue.AddToQueue(ETL_QUEUE, result_tid)
    publish_queue.Close()

def publish_error(error_url):
    publish_queue = PikaQueue(RABBIT_HOST, RABBIT_PORT, RABBIT_ACCOUNT, RABBIT_PASSWORD)
    publish_queue.AddToQueue(PREDICT_ERROR_QUEUE, error_url)
    publish_queue.Close()