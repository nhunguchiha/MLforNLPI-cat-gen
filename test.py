"""
#Generating cat with different method, be careful with choice of model, you might need to mannually uncomment and comment the code ( see line 19 to 24)

Usage:
    test.py  --temperature=<temp> [--add_random] [--beam_search]

Example: Original method :  python test.py --temperature=0.7 (although we still need to set the temp it wont be used)
        Temperature control: python test.py --temperature=0.7 --add_random
        Beam search: python test.py --temperature=0.7 --beam_search (although we still need to set the temp it wont be used)

Options:
    -h --help     Show this screen.
"""
from docopt import docopt
import numpy as np 

if __name__ == '__main__':
    args = docopt(__doc__, version='Testing our cat generator!!')
    temperature =  float(args["--temperature"])
    add_random = args['--add_random']
    beam_search = args['--beam_search']
    #get ix_to_char dictionary (the dictionary is obtained when running .py)
    #if you run training_data_5.txt use this config
    # ix_to_char =  {0: 'p', 1: 'e', 2: 's', 3: ' ', 4: '/', 5: '\\', 6: '_', 7: '\n', 8: '(', 9: 'o', 10: ')', 11: '>', 12: '#', 13: '<', 14: '|', 15: '.', 16: '^', 17: 'Y', 18: '`', 19: '-', 20: "'"}
    # vocab_size = 21
    ##if you run training_data_3.txt use this config
    vocab_size = 23
    ix_to_char = {0: 'p', 1: 'e', 2: 's', 3: ' ', 4: '/', 5: '\\', 6: '_', 7: '\n', 8: '(', 9: 'o', 10: ')', 11: '>', 12: '#', 13: '<', 14: '^', 15: '|', 16: '-', 17: '.', 18: 'Y', 19: '`', 20: "'", 21: '~', 22: '*'}

    #load model
    data = np.load("rnn_model.npz")
    Wxh = data['Wxh']
    Whh = data['Whh']
    Why = data['Why']
    bh = data['bh']
    by = data['by']

    hidden_size = 200
    h = np.zeros((hidden_size,1))


    def sample(h, seed_ix, n, temperature=temperature, add_random = add_random) :
      """
      sample a sequence of integers from the model
      this function is used when we want to use original sampling method or temperature control sampling method
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
        if add_random: #if in random mode
          ix = np.random.choice(range(vocab_size), p=p.ravel())  #sampling from distribution with higher probability to pick chars with high ps
        else: 
          ix = p.ravel().argmax() #pick the highest probability one
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
        if ix == 1:
          break
      return ixes


    def sample_beam_search(h, seed_ix, n, temperature=2, beam_width =4) :
      
      """
      sample a sequence of integers from the model
      this function is used when we want to use beam search sampling method
      h is memory state, seed_ix is seed letter for first time step
      """
      x = np.zeros((vocab_size, 1))
      x[seed_ix] = 1
      ixes = []
      # Initialize the beam with the seed input
      beam = [(h, [seed_ix], 0)]  # (hidden state, sequence, cumulative log probability)

      for t in range(n):
            new_beam = []
            for h, seq, cum_log_prob in beam:
                x = np.zeros((vocab_size, 1))
                x[seq[-1]] = 1
                # Update the hidden state
                h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
                # Compute the output vector
                y = np.dot(Why, h) + by
                # Adjust probabilities with temperature
                y = y / temperature
                # Compute the probability distribution
                p = np.exp(y) / np.sum(np.exp(y))
                # Get the top beam_width probabilities and their indices
                top_indices = np.argsort(p.ravel())[::-1][:beam_width]
                top_probs = np.sort(p.ravel())[::-1][:beam_width]
  
                # Expand each beam
                for i in range(beam_width):
                    new_seq = seq + [top_indices[i]]
                    new_cum_log_prob = cum_log_prob + np.log(top_probs[i])
                    new_beam.append((h, new_seq, new_cum_log_prob))
            
            # Keep the top beam_width sequences
            new_beam = sorted(new_beam, key=lambda x: x[2], reverse=True)[:beam_width]
            
            # Update the beam
            beam = new_beam
            
      best_seq = [beam[0][1], beam[1][1], beam[2][1], beam[3][1]]

      return best_seq 

  #Generating cat based on argument we set in command: 
    if beam_search:
          for seq in sample_beam_search(h,2,150):
              print('----\n')
              char_list = []
              for ix in seq[1:-1]:
            
                  if ix == 1:
                      break
                  char_list.append(ix_to_char[ix])
              print(''.join(char_list))
              
    else:
          print(''.join([ix_to_char[ix] for ix in  sample(h,2,150)[:-1]]))

