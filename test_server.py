import eventlet
import socketio

sio = socketio.Server(ping_timeout=60)
app = socketio.WSGIApp(
    sio,
    static_files={
        "/": "./client/dist/index.html",
        "/favicon.png": "./client/dist/favicon.png",
        "/assets": "./client/dist/assets",
    },
)

eventlet.wsgi.server(eventlet.listen(("localhost", 5000)), app)
