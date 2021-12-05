import nltk
from nltk.corpus import stopwords

stopwords = stopwords.words("english")
useless_words = ["pair", "something"]
useless_words.extend(stopwords)


def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
        yield subtree.leaves()


def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword. We can increase the length if we want to consider large phrase"""
    accepted = word not in useless_words
    return accepted


def get_terms(tree):
    for leaf in leaves(tree):
        term = [(w, p) for w, p in leaf if acceptable_word(w)]
        if len(term) > 0:
            yield term


def extractor(text):

    text = text.lower()

    sentence_re = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'
    grammar = r"""
        NBAR:
            {<NN.*|JJ|VBD|VBN>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

        NP:
            {<NN.*|JJ|VBD|VBN>+<CC><NN.*|JJ|VBD|VBN>*<NBAR>}
            {<NBAR><IN><DT><NBAR><NN.*|JJ|VBD|VBN>?}
            {<NBAR><VBD|VBN>}
            {<NBAR>}
    """
    chunker = nltk.RegexpParser(grammar)
    toks = nltk.regexp_tokenize(text, sentence_re)
    postoks = nltk.tag.pos_tag(toks)
    tree = chunker.parse(postoks)

    terms = get_terms(tree)
    return terms


if __name__ == "__main__":
    texts = []
    texts.append(
        "The man has on a light colored t-shirt with dark pants, and light sneakers. he has a large black backpack and glasses."
    )
    texts.append(
        "THE YOUNG GIRL IS WEARING LIGHT BLUE SKINNY JEANS AND A GRAY TUNIC TOP. IN HER SHOES SHE HAS NEON GREEN STRINGS. SHE HAS A BACKPACK IN THE FRONT."
    )
    texts.append(
        "A woman wearing a yellow shirt, a pair of white and black floral print pants and a pair of white and black shoes."
    )
    texts.append(
        "A girl with shoulder length hair wearing a light grey sweater with lime green undershirt with short shorts and carrying a lime green backpack."
    )
    texts.append(
        "He is wearing black shoes, khaki shorts, and a maroon polo short sleeved shirt with the collar unbuttoned. He is carrying a white bag in his left hand."
    )
    texts.append(
        "This guy has short black hair and thick black glasses. He has a black and grey striped hooded sweatshirt on and is wearing a pair of black jeans. He is also wearing a pair of white sneakers and is carrying a black bag on his left arm."
    )
    texts.append(
        "The woman is wearing black shoes with white laces. She has on dark colored pants and a blue shirt. She is running."
    )
    texts.append(
        "The man is wearing a black t-shirt and dark blue jeans, along with a silver necklace. He has his hands clasped together in front of him."
    )
    texts.append(
        "The woman is wearing a sleeveless top and shorts. She is looking down at something in her hands. She is wearing flat shoes."
    )
    texts.append(
        "The girl is wearing a red skirt, a red and white shirt and white shoes and socks."
    )
    texts.append(
        "A man steps forward with his left leg, carries a curved black backpack on his back, and has unfolding rolled paper in his right hand. He wears a short-sleeve white shirt, black pants and white shoes with his left heel on the ground."
    )
    texts.append(
        "A man visible from the side is wearing a dark short sleeve shirt, blue shorts with a ehite stripe, and white sneakers."
    )
    texts.append(
        "The woman has black hair pulled into a ponytail and is wearing a peach colored shirt with slits in the back, blue jean shorts, and blue, white and pink tennis shoes."
    )
    texts.append(
        "The lady wears a brown jacket over a red shirt blue jean shorts beige high heeled shoes she carries a small brown shoulder bag"
    )

    for text in texts:
        print("-" * 100)
        print(text)
        for term in extractor(text):
            print(term)
