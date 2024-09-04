"""

Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
Modified by Hoa Quynh Nhung Nguyen

Usage:
    my-code-2.py <input_file_train> <input_file_eval> --hidden_size=NUM --num_epochs=NUM --lr=<lr> --temperature=<temp> [--eval_mode]

Note : Pass eval_mode if you want to train model with validation loop
 
Example : python my-code-2.py training_data_3.txt validation_data_2.txt --hidden_size=200 --num_epochs=700 --lr=0.02 --temperature=0.5


Options:
    -h --help     Show this screen.
"""
print("Script started")

import numpy as np
from docopt import docopt
import sys
import random


def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
    
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n, temperature=1.0, add_random = True) :
  """
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    y = y / temperature
    p = np.exp(y) / np.sum(np.exp(y))
    if add_random:
      ix = np.random.choice(range(vocab_size), p=p.ravel())  #sampling from distribution with higher probability to pick chars with high ps
    else: 
      ix = p.ravel().argmax()
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
    if ix == 1:
      break
  return ixes

def read_file(input_file):
  """
  read input in txt file
  """
  data = open(input_file, 'r').read()
  data =  data.split('\n\n')
  return data


class Lang():
    """
    use to get char to ix and ix to char dictionary. 
    """
    def __init__(self, corpus, special_tokens=[]):
        self.char_to_ix = self.get_vocab(corpus, special_tokens)
        self.ix_to_char = {v:k for k, v in self.char_to_ix.items()}
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence:
                if w not in output:
                    output[w] = i
                    i += 1
        return output

def pad_inputs(inputs, max_length, padding_value=0):
    """
    use to pad input to the maximum length
    """
    pad_inputs = []
    for item in inputs:
        if len(item) < max_length:
            pad_input = item + [padding_value] * (max_length - len(item))
        else:
            pad_input = item
        pad_inputs.append(pad_input)
    return pad_inputs

def eval_loop(eval_data):
    """
    evaluate model with validation data, return validation loss
    """
    loss_array = []
    number_of_tokens = []
    for i in eval_data: 
      hprev = np.zeros((hidden_size,1))
      x = i[:-1]
      y = i[1:]
      xs, hs, ys, ps = {}, {}, {}, {}
      hs[-1] = np.copy(hprev)
      loss = 0
      for t in range(len(x)):
        xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
        xs[t][x[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
        ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
        loss += -np.log(ps[t][y[t],0]) # softmax (cross-entropy loss)
      loss_array.append(loss)
      number_of_tokens.append(len(x))
    eval_loss = sum(loss_array) / sum(number_of_tokens)
    return eval_loss
    
   

if __name__ == '__main__':
    args = docopt(__doc__, version='The amazing cat')
    print(args)
    #parsing argument
    input_file_train = args["<input_file_train>"]
    input_file_eval = args["<input_file_eval>"]
    hidden_size = int(args["--hidden_size"])    # size of hidden layer of neurons
    num_epochs = int(args["--num_epochs"])      # number of steps to unroll the RNN for
    learning_rate = float(args["--lr"])
    temperature =  float(args["--temperature"])
    eval_mode = args['--eval_mode']

    #read input
    train_raw = read_file(input_file_train)
    eval_raw = read_file(input_file_eval)

    #preprocessing input
    lang = Lang(train_raw, ["p", "e", "s"])
    char_to_ix = lang.char_to_ix
    ix_to_char = lang.ix_to_char
    vocab_size = len(char_to_ix)
    start_token = 's'
    eos_token = 'e'
    pad_token = 'p'
    add_eos_sos_train = [ start_token + i + eos_token for i in train_raw ]
    inputs_train = [[char_to_ix[ch] for ch in item] for item in add_eos_sos_train]
    add_eos_sos_eval = [ start_token + i + eos_token for i in eval_raw ]
    inputs_eval = [[char_to_ix[ch] for ch in item] for item in add_eos_sos_eval]
    max_length = max(len(item) for item in inputs_train)

    inputs_train = pad_inputs(inputs_train, max_length) #pad input to the max length
    inputs_eval = pad_inputs(inputs_eval, max_length) #pad input to the max length
    seq_length = max_length

    print('data has, %d unique.' % (vocab_size))
    # model parameters
    Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
    Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
    Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
    bh = np.zeros((hidden_size, 1)) # hidden bias
    by = np.zeros((vocab_size, 1)) # output bias

    n, p = 0, 0
    mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
    best_loss = 10000000000
    train_losses = []
    eval_losses = []
    epoch_count = 0

    #start training 
    while epoch_count < num_epochs:
      if epoch_count > 0 and epoch_count % 10 == 0:
          print("EPOCH",epoch_count)
          hprev = np.zeros((hidden_size,1)) # reset RNN memory
          sample_ix = sample(hprev, 2 , 150, temperature)
          txt = ''.join(ix_to_char[ix] for ix in sample_ix[1:-1])
          print('----\n %s \n----' % (txt, ))
          print('iter %d, loss: %f' % (n, train_loss)) # print progress
          print('iter %d, loss: %f' % (n, eval_loss)) # print progress
          print('----\n')
      loss_array = []
      number_of_tokens = []
      for i in random.sample(inputs_train, len(inputs_train)):
        n += 1 # iteration counter
        hprev = np.zeros((hidden_size,1)) # reset RNN memory
        x = i[:-1]
        y = i[1:]
      # forward seq_length characters through the net and fetch gradient
      #print(hprev)
        loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(x, y, hprev)
        # smooth_loss = smooth_loss * 0.999 + loss * 0.001
        loss_array.append(loss)
        number_of_tokens.append(len(x))

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                      [dWxh, dWhh, dWhy, dbh, dby],
                                      [mWxh, mWhh, mWhy, mbh, mby]):
          mem += dparam * dparam
          param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
        train_loss = sum(loss_array)/sum(number_of_tokens)

      #save loss of this epoch
      train_losses.append(train_loss)
      eval_loss = eval_loop(inputs_eval)
      if eval_loss < best_loss:
         best_loss = eval_loss
         #save model with the best validation loss if eval_mode = True.
         if eval_mode:
            np.savez('rnn_model.npz', Wxh=Wxh, Whh=Whh, Why=Why, bh=bh, by=by, ix_to_char = ix_to_char, char_to_ix = char_to_ix)

      eval_losses.append(eval_loss)
      #next epoch
      epoch_count+=1

if not eval_mode:
  np.savez('rnn_model.npz', Wxh=Wxh, Whh=Whh, Why=Why, bh=bh, by=by, ix_to_char = ix_to_char, char_to_ix = char_to_ix)
print(ix_to_char)

#Visualize traning loss and validation loss over epochs
import matplotlib.pyplot as plt
plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
plt.title('Train and Dev Losses')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.plot( [i for i in range(num_epochs)], train_losses, label='Train loss')
plt.plot([i for i in range(num_epochs)], eval_losses, label='Dev loss')
plt.legend()
plt.show()