from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import json
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

def get_training_data():
    computers = ['viga','duna','delta']
    out = {}
    for computer in computers:
        f = open('/net/vena/scratch/Dropbox/GIT/DeepLearning/losses/{0}_loss.json'.format(computer), 'r')
        out[computer] = f.read()
        if (out[computer][-2] == '"' or out[computer][-2] != "]"):
            out[computer] += ']'
        f.close()

        f = open('/net/vena/scratch/Dropbox/GIT/DeepLearning/losses/{0}_loss_batch.json'.format(computer), 'r')
        out['{0}_batch'.format(computer)] = f.read()
        if (out['{0}_batch'.format(computer)][-2] == '"' or out['{0}_batch'.format(computer)][-2] != "]"):
            out['{0}_batch'.format(computer)] += ']'
        f.close()

    return json.dumps(out)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/publish/epoch/end/', methods=['POST'])
def new_epoch():        
    socketio.emit('update', {'data': get_training_data()}, namespace='/')    
    return "OK"

@socketio.on('connect')
def test_connect():
    # need visibility of the global thread object
    print('Client connected')

    socketio.emit('update', {'data': get_training_data()}, namespace='/')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@socketio.on('message')
def handle_message(msg):
    emit('update', {'data': 'got it!'})

if (__name__ == '__main__'):
    socketio.run(app)
