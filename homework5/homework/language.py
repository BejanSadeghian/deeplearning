from .models import LanguageModel, AdjacentLanguageModel, Bigram, load_model
from . import utils

import torch
from torch.distributions.categorical import Categorical
import string

import numpy as np

def log_likelihood(model: LanguageModel, some_text: str):
    """
    Your code here

    Evaluate the log-likelihood of a given string.

    Hint: utils.one_hot might come in handy

    :param model: A LanguageModel
    :param some_text:
    :return: float
    """
    start = ''
    
    prob_matrix = model.predict_all(start)[:,-1].view((-1,1))
    for i in range(len(some_text)-1):
        start += some_text[i]
        prob_matrix = torch.cat((prob_matrix, model.predict_all(start)[:,-1].view((-1,1))), axis=1)
#    prob_matrix = model.predict_all(some_text)[:,:-1]

    str_one_hot = utils.one_hot(some_text)
    probabilities = prob_matrix * str_one_hot
    return probabilities.sum()


def sample_random(model: LanguageModel, max_length: int = 100):
    """
    Your code here.

    Sample a random sentence from the language model.
    Terminate once you reach a period '.'

    :param model: A LanguageModel
    :param max_length: The maximum sentence length
    :return: A string
    """
    vocab = string.ascii_lowercase + ' .'
    out_str = ''
    for i in range(max_length):
        prob_matrix = model.predict_all(out_str)
        dist = Categorical(np.exp(prob_matrix[:,-1]))
        new_char = vocab[dist.sample().item()]
        out_str += new_char
        if new_char == '.':
            break
    return out_str



class TopNHeap:
    """
    A heap that keeps the top N elements around
    h = TopNHeap(2)
    h.add(1)
    h.add(2)
    h.add(3)
    h.add(0)
    print(h.elements)
    > [2,3]

    """
    def __init__(self, N):
        self.elements = []
        self.N = N

    def add(self, e):
        from heapq import heappush, heapreplace
        if len(self.elements) < self.N:
            heappush(self.elements, e)
        elif self.elements[0] < e:
            heapreplace(self.elements, e)


def beam_search(model: LanguageModel, beam_size: int, n_results: int = 10, max_length: int = 100, average_log_likelihood: bool = True):
    """
    Your code here

    Use beam search for find the highest likelihood generations, such that:
      * No two returned sentences are the same
      * the `log_likelihood` of each returned sentence is as large as possible

    :param model: A LanguageModel
    :param beam_size: The size of the beam in beam search (number of sentences to keep around)
    :param n_results: The number of results to return
    :param max_length: The maximum sentence length
    :param average_log_likelihood: Pick the best beams according to the average log-likelihood, not the sum
                                   This option favors longer strings.
    :return: A list of strings of size beam_size
    """
    from heapq import heappush, heapreplace
    import heapq
    import operator
    beam_size = n_results
    
    def get_next_letters(string, beam_size, model, vocab):
        next_prob = np.exp(model.predict_all(string)[:,-1])
        return list(zip(*heapq.nlargest(beam_size, enumerate(next_prob), key=operator.itemgetter(1))))[0], next_prob
    
    def get_ll(text, model, average_log_likelihood):
        # print(log_likelihood(model, text) / len(text))
        ll = log_likelihood(model, text) / len(text) if average_log_likelihood else log_likelihood(model, text)
        return (text, ll)
        
    vocab = string.ascii_lowercase + ' .'    
#    h = TopNHeap(beam_size)
    #Prime the heap
    init_letters, probabilities = get_next_letters('', beam_size, model, vocab)
    track = []
    for i in init_letters:
#        h.add(probabilities[i])
        track.append(get_ll(vocab[i], model, average_log_likelihood))
    end_bool = False
    for i in range(max_length):
        # print(i,'d')
        period_counter = 0
        iteration = []
        for t in track:
            e = t[0] #the string
            if e[-1] == '.':
                iteration.append(t)
                period_counter  += 1
                continue
            else:
                # test_nodes, probabilities = get_next_letters(e, beam_size, model, vocab)
                # iteration.append([get_ll(e + vocab[new_char], model, average_log_likelihood) for new_char in test_nodes])  
                for letter in vocab:
                    test_e = e+letter
                    iteration.append( get_ll(test_e, model, average_log_likelihood))

            if period_counter == beam_size:
                end_bool = True

        if end_bool:
            break
        else:
            iteration = list(set(iteration))
            # print(track)
            # print(iteration)
            track = sorted(iteration, key=lambda x: x[1], reverse=True)[:beam_size]
            # print(track)
            # input()
    track = sorted(track, key=lambda x: x[1])
    return [x for x,y in track][:n_results]
            
            
                
                
    
    
#    raise NotImplementedError('beam_search')


if __name__ == "__main__":
    """
      Some test code.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', choices=['Adjacent', 'Bigram', 'TCN'], default='Adjacent')
    args = parser.parse_args()

    lm = AdjacentLanguageModel() if args.model == 'Adjacent' else (load_model() if args.model == 'TCN' else Bigram())

    for s in ['abcdefg', 'abcgdef', 'abcbabc', '.abcdef', 'fedcba.']:
        print(s, float(log_likelihood(lm, s)))
    print()

    for i in range(10):
        s = sample_random(lm)
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100):
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100, average_log_likelihood=True):
        print(s, float(log_likelihood(lm, s)) / len(s))
