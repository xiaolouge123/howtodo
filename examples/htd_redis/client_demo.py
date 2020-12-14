import redis

pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
r_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

r_client.set('testkey1', '1234')
print(r_client.get('testkey1'))
r_client.set('testkey2', 123)
print(r_client.get('testkey2'))