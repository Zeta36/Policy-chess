import fnmatch
import os
import numpy as np
import chess.pgn


def find_files(directory, pattern='*.pgn'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_text(directory):
    '''Generator that yields text raw from the directory.'''
    files = find_files(directory)
    text = " "
    for filename in files:
        k = 0
        pgn = open(filename)
        for offset, headers in chess.pgn.scan_headers(pgn):
            pgn.seek(offset)
            game = chess.pgn.read_game(pgn)
            node = game
            while not node.is_end():
                next_node = node.variation(0)
                label_san = node.board().san(next_node.move)
                if " " + label_san + " " not in text:
                    text += label_san + " "
                node = next_node
            if k % 100 == 0 and k > 1:
                print ("Labeling file: " + filename + ", step: " + str(k))
            k += 1
        pgn.close()
    y = []
    for index, item in enumerate(text):
        y.append(text[index])
    y = np.array(y)
    np.savetxt("labels.txt", y.reshape(1, y.shape[0]), delimiter="", newline="\n", fmt="%s")


def main():
    load_generic_text("./datasets")
    print ("Done.")

if __name__ == '__main__':
    main()