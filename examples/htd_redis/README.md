install docker 

pull redis latest image
`docker pull redis:latest`

run redis container and redis server
`docker run -itd --name redis -p 127.0.0.1:6379:6379 redis`

run redis cli in container
`docker exec -it redis redis-cli`