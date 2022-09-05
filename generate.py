import argparse
import json

import numpy as np

parser = argparse.ArgumentParser(description='Get text from model')
parser.add_argument('-m', '--model', type=str, help='путь к файлу, из которого загружается модель.',
                    default="all_grams.txt")
parser.add_argument('-pre', '--prefix', nargs='+',
                    help='необязательный аргумент. Начало предложения (одно или несколько слов). Если не указано, '
                         'выбираем начальное слово случайно из всех слов.', default=None)
parser.add_argument('-len', '--length', type=int, help='длина генерируемой последовательности.', default=20)
args = parser.parse_args()


def separate_grams(all_grams):
    one_grams = {}
    two_grams = {}
    three_grams = {}
    for gram in all_grams.keys():
        if len(gram.split(' ')) == 1:
            one_grams[gram] = all_grams[gram]
        elif len(gram.split(' ')) == 2:
            two_grams[gram] = all_grams[gram]
        elif len(gram.split(' ')) == 3:
            three_grams[gram] = all_grams[gram]

    return one_grams, two_grams, three_grams


def make_text(all_grams, length, prefix=None):
    one_grams, two_grams, three_grams = separate_grams(all_grams)
    words = one_grams.keys()
    words = list(words)
    if prefix is None or prefix == "":
        arr = np.random.choice(list(two_grams.keys()), 1)[0].split(' ')
    else:
        arr = prefix
    arr = list(arr)
    for j in range(length):

        p_arr = []
        for word in words:
            if arr[-2] + " " + arr[-1] + " " + word in three_grams.keys():
                p_arr.append(three_grams[arr[-2] + " " + arr[-1] + " " + word] / two_grams[arr[-2] + " " + arr[-1]])
            else:
                p_arr.append(0)

        # пару раз была ошибка из-за точности вычислений, поэтому нормализую
        p_arr /= np.sum(p_arr)
        arr.append(np.random.choice(words, 1, p=p_arr)[0])
    print(" ".join(arr))


if __name__ == '__main__':
    model = args.model
    with open(model, 'r', encoding='utf-8') as file:
        all_grams = json.load(file)

    length = args.length
    prefix = args.prefix
    make_text(all_grams, length, prefix)
