from hashlib import sha256
import os
import json
BLOCKCHAIN_DIR = 'blockchain/'



def updatehash(*args):
    hashing_text = ""; h = sha256()

    #loop through each argument and hash
    for arg in args:
        hashing_text += str(arg)

    h.update(hashing_text.encode('utf-8'))
    return h.hexdigest()


class Block: 
    
    def __init__(self, data = None, number=0, previous_hash = "0" * 64, nonce=0):
        self.data = data
        self.number = number
        self.previous_hash = previous_hash
        self.nonce = nonce

    def hash(self):
        return updatehash(
            self.number,
            self.previous_hash,
            self.data,
            self.nonce
        )

    def __str__(self):
        return str("Block#: %s\nHash: %s\nPrevious: %s\nData: %s\nNonce: %s\n" %(
            self.number,
            self.hash(),
            self.previous_hash,
            self.data,
            self.nonce
            )
        )

class Blockchain:   
    difficulty = 2 

    def __init__(self):
        self.chain = []
    
    def add(self, block):
        block_data = {
            'block_no' : block.number,
            'block_hash' : block.hash(),
            'previous_hash' : block.previous_hash,
            'nonce' : block.nonce,
            'data' : block.data,
        }

        with open(BLOCKCHAIN_DIR +  str(block.number), 'w') as f:
            json.dump(block_data, f, indent=4, ensure_ascii=False)
            f.write('\n')

        self.chain.append(block)


    def remove(self, block):
        self.chain.remove(block)

    def mine(self, block):
        try:
            block.previous_hash = self.chain[-1].hash()
        except IndexError:
            pass

        while True:
            if block.hash()[:self.difficulty] == "0" * self.difficulty:
                self.add(block); break
            else:
                block.nonce +=1

    def isValid(self):   
        for i in range(1, len(self.chain)):
            _previous = self.chain[i].previous_hash
            _current = self.chain[i-1].hash()
            if _previous != _current or _current[:self.difficulty] != "0"*self.difficulty:
                return False
        return True


    def getChain(self):
        current_chain = [] 
        current_chain = self.chain
        return current_chain

def main():
    blockchain = Blockchain()

    database = ["hello 1", "hello 2", "hello 3", "hello 4"]

    num = 0
    for data in database:
        num += 1
        blockchain.mine(Block(data, num))

    crCahin = blockchain.getChain()
    for e in crCahin:
        print(e)
    
    # # Adding a new block
    # num += 1
    # blockchain.mine(Block('hello 5', num))

    # for block in blockchain.chain:
            #print(block)



    #manually corrupt the blockchain
    # blockchain.chain[2].data = "new data"
    
    #printing the validity state of the blockchain
    #print(blockchain.isValid())

if __name__ == '__main__':
    main()