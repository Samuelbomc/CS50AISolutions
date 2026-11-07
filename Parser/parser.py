import nltk
import sys
from nltk import Tree

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP
NP -> N | Det NP | AP NP | N PP | N Adv | N VP
VP -> V | V NP | V AvP | C VP 
PP -> P NP | P C | P AP
AP -> Adj | Adj AP | Adj NP | Det Adj NP 
AvP -> Adv | Adv VP
C -> Conj NP | Conj VP | Conj
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    tokens = nltk.word_tokenize(sentence)
    for i, token in enumerate(tokens):
        if not any(char.isalpha() for char in token):
            tokens.remove(token)
            continue
        tokens[i] = tokens[i].lower()
    return tokens


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees (at any depth).
    """
    def contains_np(subtree):
        """
        Check recursively if a subtree contains any NP-labeled node.
        """
        for descendant in subtree.subtrees():
            if descendant != subtree and descendant.label() == "NP":
                return True
        return False

    noun_chunks = []
    
    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            if not contains_np(subtree):
                noun_chunks.append(subtree)
    
    return noun_chunks


if __name__ == "__main__":
    main()
