from datetime import datetime
from hashlib import sha256
import threading
import socket
import time

HOST = "localhost"
DATETIME_FORMAT = "%d/%m/%Y %H:%M:%S"


def time_now():
    return datetime.now().strftime(DATETIME_FORMAT)


class Block:
    def __init__(self, data, previous_hash, timestamp):
        self.data = data
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        return sha256(self.__dict__.__str__().encode()).hexdigest()


class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.difficulty = '000'

    @staticmethod
    def create_genesis_block():
        return Block("genesis_block", "0", time_now())

    def add_block(self, data):
        previous_block = self.chain[-1]
        new_block = Block(data, previous_block.hash, time_now())
        self.chain.append(new_block)

    def proof_of_work(self, previous_proof):
        new_proof = 1
        check_proof = False

        while check_proof is False:
            new_try = str(new_proof ** 2 - previous_proof ** 2).encode()
            hash_operation = sha256(new_try).hexdigest()
            if hash_operation[:len(self.difficulty)] == self.difficulty:
                check_proof = True
            else:
                new_proof += 1

        return new_proof

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash():
                return False

            if current_block.previous_hash != previous_block.hash:
                return False

        return True


class Peer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.peer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.peer_socket.bind((self.host, self.port))
        self.peer_socket.listen()

    def receive_message(self):
        while True:
            client_socket, client_address = self.peer_socket.accept()
            message = client_socket.recv(1024).decode()
            print(f"\nReceived message: {message}")
            client_socket.close()
            time.sleep(1)

    def send_message(self):
        while True:
            try:
                target_port, msg = input("\nInsert the target port and the message separated by << : ").split("<<")
                target_port = int(target_port)
                target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                target_socket.connect((HOST, target_port))
                target_socket.send(msg.encode())
                target_socket.close()
            except ValueError:
                print("Incorrect input")
            except Exception as e:
                print("Incorrect port")


def main():
    port = int(input("Provide a desired port for this node: "))
    node = Peer("localhost", port)

    print(f"Node created at port: {port}")
    print(f"You can now receive and send messages to other nodes.")

    threading.Thread(target=node.receive_message).start()
    threading.Thread(target=node.send_message).start()


if __name__ == "__main__":
    main()
