# 🐱 Cat Generator with RNNs 🐱

Welcome to the world of AI-generated cats! This project is an attempt to teach a character-based Recurrent Neural Networks (RNNs) to create glorious (or at least recognizable) ASCII cats. Spoiler: It's harder than you'd think. 😸

## What's This About?
I wanted to see if RNNs could generate unique cat figures after learning from a few examples. Turns out, RNNs are great at memorizing cats but not so great at being creative. So I went on a journey to teach the model how to generalize and dream up new feline friends.

## The Results 🐾
**Cats Generated?** Sort of.

**Creative Cats?** Occasionally.

**Novel Cats?** Uhh… let’s say they tried.

## Try It Yourself!
Clone the repo, train some models, and watch your AI try its best to draw a cat. It might not be Picasso, but it could give you a laugh!

### The data
You have two version of training data to play with : training_data_3.txt include different cat type and training_data_4.txt include more uniform cats.

There is also one file of validation data named validation_data_2.txt

### Running the model

Run my-code-2.py to train the model. It needs various parameters, in this form:

```bash
python my-code-2.py training_data_3.txt validation_data_2.txt --hidden_size=200 --num_epochs=700 --lr=0.02 --temperature=0.5

If you'd like to add validation loop, try :

python my-code-2.py training_data_3.txt validation_data_2.txt --hidden_size=200 --num_epochs=700 --lr=0.02 --temperature=0.5 --eval_mode

### Sampling your trained model

After training, run test.py to load the model and test its generative ability. There 3 sampling strategies.
Example usage: 
Original method  (although we still need to set the temp it wont be used):  
python test.py --temperature=0.7
Temperature control: 
python test.py --temperature=0.7 --add_random
Beam search: python test.py (again although we still need to set the temp it wont be used)
--temperature=0.7 --beam_search 
