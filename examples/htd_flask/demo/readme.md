when use gunicorn with flask app defined in a ApiServer class, like in app.
we need to get the app from the server 

gunicorn demo.wsgi:app -b 127.0.0.1:8080 -w 4