# main.py
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import cv2

app = Flask(__name__)
socketio = SocketIO(app)
model = YOLO('best.pt')
model_classify = YOLO('best classify.pt')
crack = 12
noncrack = 78

def clasify_crack():
    cap = cv2.VideoCapture("wallvid.mp4")
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        results = model(frame)
        for box in results[0].boxes.xywh :
            x, y, w, h = box
            cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (25, 0, 255), 2)
            cv2.putText(frame, "Crack", (int(x-w/2) + 5, int(y-h/2) + 20), cv2.FONT_HERSHEY_SIMPLEX, .65, (25, 0, 255), 2, cv2.LINE_AA)

        results_classify = model_classify(frame)
        crack = results_classify[0].probs.data.numpy()[0]
        noncrack = results_classify[0].probs.data.numpy()[1]

        if crack > noncrack:
            cv2.putText(frame, "Crack", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (25, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Non Crack", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (25, 255, 0), 2, cv2.LINE_AA)
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        socketio.emit('update_crack', int(crack * 100), namespace='/crack')
        socketio.emit('update_noncrack', int(noncrack * 100), namespace='/crack')
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template('vid.html')

@app.route('/video_feed')
def video_feed():
    return Response(clasify_crack(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect', namespace='/crack')
def connect_crack():
    print('connect crack')

@socketio.on('update_crack', namespace='/crack')
def handle_update_crack(json):
    emit('update_crack', crack, broadcast=True)

@socketio.on('update_noncrack', namespace='/crack')
def handle_update_noncrack(json):
    emit('update_noncrack', noncrack, broadcast=True)

if __name__ == '__main__':
    model.predict('crack/7081-9.jpg')
    model_classify.predict('crack/7081-9.jpg')
    socketio.run(app, debug=True)
