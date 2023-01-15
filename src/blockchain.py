from torchvision.transforms import transforms
from hashlib import sha256
from PIL import Image
import torch
import json

from src.model import HistEqualization, SmoothImage


class QualityAssessmentModel:
    def __init__(self, model_path, device='cpu'):
        self.model = torch.jit.load(model_path)
        self.model = self.model.to(device)
        self.model = self.model.eval()
        self.pre_process = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((224, 224)),
            HistEqualization(),
            SmoothImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.25])
        ])

    def predict(self, Xi):
        Xi = self.pre_process(Image.fromarray(Xi))
        batched_image = torch.FloatTensor(Xi).unsqueeze(0)
        output = torch.argmax(torch.softmax(self.model(batched_image).detach()[0], dim=0)).item()
        print(output)
        return bool(output)


class Block:
    def __init__(self, idx, transactions, time_stamp, previous_hash):
        self.idx = idx
        self.transactions = transactions
        self.time_stamp = time_stamp
        self.previous_hash = previous_hash
        self.hash = None
        self.nonce = 0

    def compute_hash(self):
        block = json.dumps(self.__dict__.__str__(), sort_keys=True).encode()
        return sha256(block).hexdigest()


class Transaction:
    def __init__(self, sender, data, data_name):
        self.sender = sender
        self.data = data
        self.data_name = data_name


class Blockchain:
    def __init__(self, model_path, get_time):
        self.get_time = get_time
        self.difficulty = 2
        self.unordered_transactions = []
        self.chain = [self.__create_genesis_block()]
        self.quality_model = QualityAssessmentModel(model_path, 'cpu')

    def __create_genesis_block(self):
        genesis_block = Block(0, [], self.get_time(), '0')
        genesis_block.hash = genesis_block.compute_hash()
        return genesis_block

    def __is_valid_proof(self, block, proof):
        starts_with = proof.startswith('0' * self.difficulty)
        is_equal_hash = proof == block.compute_hash()
        return starts_with and is_equal_hash

    def __proof_of_work(self, block):
        block.nonce = 0

        computed_hash = block.compute_hash()
        while not computed_hash.startswith('0' * self.difficulty):
            block.nonce += 1
            computed_hash = block.compute_hash()

        return computed_hash

    def add_block(self, block, proof):
        previous_hash = self.chain[-1].hash

        if previous_hash != block.previous_hash:
            return False
        if not self.__is_valid_proof(block, proof):
            return False

        block.hash = proof
        self.chain.append(block)
        return True

    def add_transaction(self, transaction):
        self.unordered_transactions.append(transaction)

    def mine(self):
        if not self.unordered_transactions:
            return False

        quality_passed_transactions = []
        for transaction in self.unordered_transactions:
            if self.quality_model.predict(transaction.data):
                quality_passed_transactions.append(transaction)

        last_block = self.chain[-1]

        new_block = Block(idx=last_block.idx + 1,
                          transactions=quality_passed_transactions,
                          time_stamp=self.get_time(),
                          previous_hash=last_block.hash)

        proof = self.__proof_of_work(new_block)
        self.add_block(new_block, proof)

        self.unordered_transactions = []
        if self.difficulty < 9:
            self.difficulty += 1

        return new_block.idx
