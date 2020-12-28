from flask import Flask, Response

class ApiServer():
    def __init__(self):
        self.host='localhost'
        self.port=8080
        self.app = Flask('demo')
        self.setup_route()
    
    def start_server(self):
        self.app.run(host=self.host, port=self.port, debug=True)

    @staticmethod
    def index():
        return Response(
            'ok',
            200
        )

    def setup_route(self):
        self.app.add_url_rule('/', 'index', self.index)

def main():
    api = ApiServer()
    api.start_server()

if __name__=="__main__":
    main()
else:
    pass
    # run this code when this file is imported