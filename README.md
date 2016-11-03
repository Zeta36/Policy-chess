# A Policy Network in Tensorflow to classify chess moves

This is a TensorFlow implementation of a supervised learning in a policy network for chess moves classification.

<table style="border-collapse: collapse">
<tr>
<td>
<p>
This work is inspired in the SL policy network used by <b>Google DeepMind</b> in the program AlphaGo [AlphaGo Nature Paper](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf).
</p>
<p>
The network models the probability for every legal chess move given a chess board based only in the raw state of the game.
In this sense, the input s to the policy network is a simple representation of the board state using a tensor <code>(batc_sizex8x8x8)</code> with information of the chess board piece state, the number of the movement in the game, the current player, etc.
</p>
<p>
The SL policy network Pσ(a|s) alternates between convolutional layers with weights σ, and rectifier nonlinearities. A final softmax
layer outputs a probability distribution over all legal moves a (labels).
</p>
<p>
The policy network is trained on randomly sampled state-action pairs (s, a), using stochastic gradient ascent to
maximize the likelihood of the human move a selected in state. 
</p>
</td>
</tr>
</table>

## Preparing the Data sets

We train the 3-layer policy network using any set of chess games stored as PGN files. To prepare the training and the validation data set, we just need to download many PGN file (more data means more accuracy after the training) and put them in the datasets folder (there is in the repository some pgn examples to use).

After that, we run in the console:
```bash
python pgn-to-txt.py
```

In this way, the PGN files will be reformated in the proper way, and chuncked in a tuple of (board state, human move).
We the pgn-to-txt.py script finish, go into the datasets folder and copy almost all the "*.txt" files generated into a new folders called "data_train", and some text files into another folder called "data_validation".

Finally, you have to run 
```bash
python pgn-to-label.py
```

And we will get the labels for the SL. This labels will be generated and saved in a labels.txt file inside the "labels" folder.

## Training

Training is a easy step. Just run:
```bash
python train.py
```

You can adjust before if you wish some hyperparameters inside this python script.

## Playing

Once the model is trained (and the loss has converged), you can play a chess game against the SL policy network.
Just type:
```bash
python play.py
```

The machine moves will be generate by the policy network, and the human moves in the game will be asked to you to be type in the keyboard.
In order to move, you have to know the san <b>Algebraic notation</b> (https://en.wikipedia.org/wiki/Algebraic_notation_(chess)).

The game board is printed in ASCII, but you can use any online chess board configuration (like this http://www.apronus.com/chess/wbeditor.php) to mimic the movements so you can see clearly the game. 

## Requirements

TensorFlow needs to be installed before running the training script.
TensorFlow 0.10 and the current `master` version are supported.

In addition, [python-chess](https://github.com/niklasf/python-chess) must be installed for reading and writing PGN files, and for the play.py script to work.

## Results

After some thousands of training steps, the model is able to generalize and play a reasonable chess game based only in the prediction of the human movements in the training process.
