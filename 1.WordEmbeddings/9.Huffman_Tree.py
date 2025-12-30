# [MXNLP-1-06] 9.HuffmanTree.py
# Creating a Huffman tree
#
# This code is used in the Natural Language Processing (NLP)
# online course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/Natural-Language-Processing
#
# A detailed description of this code can be found in
# https://youtu.be/kMnPLkCn-_U
#
import numpy as np
from queue import PriorityQueue

class nodetype:
    def __init__(self, symbol, freq):
        self.symbol = symbol   # the value of a character
        self.frequency = freq  # the number of times the character is in the file
        self.left = None      
        self.right = None
    
    def __lt__(self, other):
        return self.frequency < other.frequency

# n: the number of characters in the file
# PQ: a priority queue
def huffman_tree(n, PQ):
    for _ in range(n-1):
        p = PQ.get()      # remove(PQ, p)
        q = PQ.get()      # remove(PQ, q)
        sum_freq = p.frequency + q.frequency
        r = nodetype(None, sum_freq) # new nodetype
        r.left = p
        r.right = q
        PQ.put(r)         # insert(PQ, r)
        
    r = PQ.get()          # remove(PQ, r)
    return r

chars = ['a', 'b', 'c', 'd', 'e', 'f']
freqs = [16, 5, 12, 17, 10, 25]

# Construct a priority queue for Huffman coding
PQ = PriorityQueue()
for i in range(len(chars)):
    node = nodetype(chars[i], freqs[i])
    PQ.put(node)
    
# Build a Huffman tree
root = huffman_tree(len(chars), PQ)

# Given a Huffman tree, we find Huffman code for each char.
def huffman_code(node, c=[], code={}):
    if node is not None:
        if node.symbol is None:  # not a leaf node
            huffman_code(node.left, c + [0], code)
            huffman_code(node.right, c + [1], code)
        else: # leaf node
            code[node.symbol] = c

    return code

huffman_code = huffman_code(root)
print(huffman_code)
