import requests
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, server a!'

@app.route('/jump')
def jump():
    res = requests.get('http://localhost:5001/')

    return 'Hello, jump from {}!'.format(res.text)


def main():
    app.run(host='127.0.0.1', port=5000)

if __name__=="__main__":
    main()