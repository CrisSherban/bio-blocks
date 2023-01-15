from flask import Flask, flash, request, redirect, render_template, url_for
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image
import numpy as np
import threading
import pickle
import socket
import time
import os

from src.blockchain import Blockchain, Transaction

HOST = 'localhost'
DATETIME_FORMAT = '%d/%m/%Y %H:%M:%S'
MODEL_PATH = '../models/model_torch_script.pt'
UPLOAD_FOLDER = '../datasets/uploaded'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def get_time():
    return datetime.now().strftime(DATETIME_FORMAT)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class Node:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.peer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.peer_socket.bind((self.host, self.port))
        self.peer_socket.listen()
        self.blockchain = Blockchain(MODEL_PATH, get_time)

    def receive_updates(self):
        while True:
            addr = None
            client_socket, client_address = self.peer_socket.accept()
            data = []
            while True:
                packet, addr = client_socket.recvfrom(4096)
                if not packet:
                    break
                data.append(packet)
            self.blockchain.chain = pickle.loads(b"".join(data))
            # One should first verify the chain here
            print(f"\nReceived update from {addr}")
            client_socket.close()
            time.sleep(1)

    def send_blockchain(self):
        while True:
            try:
                target_port = input("\nInsert the target port: ")
                target_port = int(target_port)
                target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                target_socket.connect((HOST, target_port))
                target_socket.send(pickle.dumps(self.blockchain.chain))
                target_socket.close()
            except Exception as e:
                print("Incorrect port")

    def add_and_mine(self, sender, data, data_name):
        new_transaction = Transaction(sender, data, data_name)
        self.blockchain.add_transaction(new_transaction)
        self.blockchain.mine()


if __name__ == '__main__':
    port = int(input('Provide a port for the new node: '))
    flask_port = int(input('Provide a port for the flask app: '))
    node = Node(HOST, port)
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.use_reloader = False


    @app.context_processor
    def inject_enumerate():
        return dict(enumerate=enumerate)


    @app.route('/get_chain', methods=['GET'])
    def get_chain():
        chain_data = []
        for block in node.blockchain.chain:
            chain_data.append(block.__dict__)
        return render_template('chain.html', chain_data=chain_data)


    @app.route('/upload', methods=['GET', 'POST'])
    def upload_file():
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                file_name = request.form.get('name')
                if not file_name:
                    file_name = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
                img = Image.open(f'{UPLOAD_FOLDER}/{file_name}')
                img = np.array(img)
                node.add_and_mine(node.port, img, file_name)

                return redirect(url_for('upload_file', name=file_name))
        return render_template('upload.html')


    threading.Thread(target=app.run, args=(HOST, flask_port, False,)).start()
    threading.Thread(target=node.receive_updates).start()
    threading.Thread(target=node.send_blockchain).start()
