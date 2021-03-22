import pika
import json
class PikaQueue():
    def __init__(self, host, port, username, password):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host,port=port,credentials=pika.PlainCredentials(username=username,password=password), heartbeat=0))
        self.channel = self.connection.channel()
        self.channel.basic_qos(prefetch_count=1)
    def DeclareQueue(self, queuename):
        self.channel.queue_declare(queue = queuename)
    def AddToQueue(self, queuename, data):
        self.channel.basic_publish(exchange = '', routing_key = queuename, body = json.dumps(data), properties = pika.BasicProperties(delivery_mode=2))
    def GetFromQueue(self, queuename, callback, auto_ack):
        self.channel.basic_consume(queue = queuename, on_message_callback = callback,auto_ack= auto_ack)
        self.channel.start_consuming()
    def Close(self):
        self.connection.close()

