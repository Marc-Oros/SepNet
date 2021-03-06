import spacy
from utils import get_word_synonyms
import re
from image_manipulation import debug
import random

class SentenceSimplifier:
    def __init__(self):
        self.synonyms = get_word_synonyms()
        self.nlp = spacy.load("en_core_web_sm")
        self.synonymDict = {}
        for item in self.synonyms:
            self.synonymDict[item[0]] = item

    """
        Simplifies a sentence
        Return values: modified sentence or not (bool), pair of simplified items (tuple), simplified sentence (string)
    """
    def simplify(self, sentence, classCombinations, shuffle=True):
        nouns = [None, None]
        baseNouns = [None, None]
        fastCheck = all(item[0] == classCombinations[0][0] for item in classCombinations)
        if len(classCombinations) == 0:
            return False, (None, None), (None, None), sentence
        for pair in classCombinations:
            pairList = list(pair)
            if shuffle is True:
                random.shuffle(pairList)
            pair = tuple(pairList)
            existsPair = [False, False]
            currSynonyms = [self.synonymDict[pair[0]], self.synonymDict[pair[1]]]
            for noun in currSynonyms[0]:
                if re.search(r"\b{}\b".format(noun), sentence) is not None:
                    existsPair[0] = True
                    nouns[0] = noun
                    baseNouns[0] = currSynonyms[0][0]
                    break
            if fastCheck is True and existsPair[0] is False:
                return False, (None, None), (None, None), sentence
            for noun in currSynonyms[1]:
                if re.search(r"\b{}\b".format(noun), sentence) is not None:
                    existsPair[1] = True
                    nouns[1] = noun
                    baseNouns[1] = currSynonyms[1][0]
                    break
            if existsPair[0] is True and existsPair[1] is True:
                break
        if existsPair[0] is False or existsPair[1] is False:
            return False, (None, None), (None, None), sentence
        doc = self.nlp(sentence)
        wordMask = [1 for i in range(len(doc))]
        for chk in doc.noun_chunks:
            if chk.root.text == nouns[0] or chk.root.text == nouns[1]:
                for word in chk:
                    if word.head.text in nouns and word.pos_ not in ['DET', 'PROPN'] and word.text != chk.root.text:
                        wordMask[word.i] = 0
        finalSentence = []
        for word, keep in zip(doc, wordMask):
            if keep == 1:
                finalSentence.append(word.text)
        finalSentence = ' '.join(finalSentence)
        if len(doc) != sum(wordMask):
            debug('Simplifier - {}'.format(nouns))
            debug('Simplifier - {}'.format(sentence))
            debug('Simplifier - {}'.format(finalSentence))
            return True, tuple(baseNouns), tuple(nouns), finalSentence
        return False, (None, None), (None, None), sentence

    def simplifyStrange(self, sentence, classCombinations):
        if len(classCombinations) != 1:
            return False, (None, None), (None, None), sentence

        return self.simplify(sentence, classCombinations, False)
