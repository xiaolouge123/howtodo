from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, server b!'

def main():
    app.run(host='127.0.0.1', port=5001)

if __name__=="__main__":
    main()