import nltk
import math
import re
import typing as T

STOPWORDS = ['ll', 's', 'd', 't', '!', "'ll", "'m", "'s", "n't", "'re", '%', '-', '"', '/', "don'", '',
    'a', 'the', 'of', 'and', 'but', 'or', 'for', 'to', 'are', 'be', 'can', 'do', 'get', 'have', 'i', "i'm",
    'in', 'is', 'it', 'me', 'my', 'on', 'that', 'you', 'your']
# punctuation = [l.strip() for l in open('data/punctuation.txt').readlines()]
FILE1 = "data/hhguide.txt"
FILE2 = "data/beowulf.txt"

# Add in punctuation, if desired
#stopwords += punctuation

def isNumber(num: str) -> bool:
    """Check if a string is a number

    Args:
        num (str): piece of text

    Returns:
        bool: True iff the string can be converted
            to a number
    """
    try:
        float(num)
        return True
    except ValueError:
        return False

def computeFreqDistribution(doc: str, stopwords: bool = False) -> nltk.FreqDist:
    """Computes the frequency of each word in a document

    Args:
        doc (str): string containing the entire document
        stopwords (bool): boolean flag indicating whether or not
            to remove the stopwords from the sentence. True indicates
            to remove the stopwords.

    Returns:
        nltk.FreqDist: frequency distribution
    """
    tokens = nltk.regexp_tokenize(doc,'\S+')
    filtered_tokens = [w.lower().strip('.,?!"\'') for w in tokens]
    consolidated_tokens = []
    for w in filtered_tokens:
        if isNumber(w):
            consolidated_tokens.append("<NUMBER>")
            continue            
        elif re.match("[\d]+(pm|am)$", w):
            consolidated_tokens.append("<TIME>")
            continue
        elif re.match("[\d]+:[\d]+(pm|am)?$", w):
            consolidated_tokens.append("<TIME>")
            continue
        elif re.match("\(?(\w+)\)?$", w):
            m = re.match("\(?(\w+)\)?$", w)
            consolidated_tokens.append(m.group(1))
            continue
        else:
            consolidated_tokens.append(w) 
    
    if stopwords:
        consolidated_tokens = [w for w in consolidated_tokens if w not in STOPWORDS and w != "" ]
    else:
        consolidated_tokens = [w for w in consolidated_tokens if w != ""]
             
    fd = nltk.FreqDist(consolidated_tokens)
    return fd

def computeUnigramDistribution(doc: str, n_words: int = None, stopwords: bool = False) -> T.Tuple[dict, float]:
    """
    Computes the relative frequencies (i.e., probs) of the most common unigrams
        in a document

    Args:
        doc (str): string containing the entire document
        n_words (int, optional): Number of most common words to consider.
            Defaults to None.
        stopwords: boolean flag indicating whether or not
            to remove the stopwords from the sentence. True indicates
            to remove the stopwords.

    Returns:
        dict: relative frequencies of the form dist[word] = prob
        float: sum of all the probabilities of the n_words most frequent unigrams
    """
    fd = computeFreqDistribution(doc, stopwords)
    keys = list(fd.keys())[:n_words]
    values = list(fd.values())[:n_words]
    N = float(sum(values))
    dist = {}
    for key in keys:
        dist[key] = float(fd[key])/N
    return (dist,N)

def mergeDistributionJS(dist1: dict, dist2: dict) -> dict:
    """
    Merges the two distributions used in the JS divergence

    Args:
        dist1 (dict): probability distribution of the form dist1[word] = prob
        dist2 (dict): probability distribution of the form dist2[word] = prob

    Returns:
        dict: New merged distribution including all words from both distributions
    """
    mergeDist = {}
    for key in dist1.keys():
        mergeDist[key] = 1/2*dist1[key]
    for key in dist2.keys():
        if key in mergeDist.keys():
            mergeDist[key] += 1/2*dist2[key]
        else:
            mergeDist[key] = 1/2*dist2[key]
    return mergeDist

def KLDivergence(P: dict, M: dict, log_base: float = math.e) -> float:
    """
    Computes the KL divergence for two distributions
        KL(P||M) = \sum_{x \in X}[p(x) * \log(p(x)/q(x))]

    Args:
        P (dict): probability distribution of words
        M (dict): probability distribution of words
        log_base (float): Base value to use for log.
            Defaults to Euler's constant

    Returns:
        float: KL divergence of two distributions
    """
    div = 0
    for key in P.keys():
        div += P[key] * math.log(P[key] / M[key], log_base)
    return div

def JSDivergence(doc1: str, doc2: str, num_words: int = None, log_base: float = math.e, stopwords: bool = False) -> float:
    """
    Calculates the JS Divergence value for two corpora

    Args:
        doc1 (str): string containing the entire document
        doc2 (str): string containing the entire document
        num_words (int): number of most frequent words to
            consider. Defaults to all words.
        log_base (float): Base value to use for log.
            Defaults to Euler's constant
        stopwords (bool): boolean flag indicating whether or not
            to remove the stopwords from the sentence. True indicates
            to remove the stopwords.

    Returns:
        float: the JS divergence of the two corpora
    """
    P, N1 = computeUnigramDistribution(doc1, num_words, stopwords)
    Q, N2 = computeUnigramDistribution(doc2, num_words, stopwords)
    M = mergeDistributionJS(P, Q)
    js = 1/2*KLDivergence(P, M, log_base) + 1/2*KLDivergence(Q, M, log_base)
    return js / math.log(log_base)

if __name__ == "__main__":
    file1 = input("Please input first file path...\n")
    file2 = input("Please input second file path...\n")
    import time
    doc1 = open(file1, 'r').read()
    doc2 = open(file2, 'r').read()
    
    print("\nRESULTS")
    print("-----------")
    start = time.time()
    js = JSDivergence(doc1, doc2)
    end = time.time()
    print(f"JS Divergence = {js}")
    print(f"Time taken = {end-start:>10.6f}")