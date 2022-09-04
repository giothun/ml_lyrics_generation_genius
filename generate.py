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
    for gram in all_grams.keys():
        if len(gram.split(' ')) == 1:
            one_grams[gram] = all_grams[gram]
        elif len(gram.split(' ')) == 2:
            two_grams[gram] = all_grams[gram]

    return one_grams, two_grams


def make_text(all_grams, length, prefix=None):
    one_grams, two_grams = separate_grams(all_grams)
    words = one_grams.keys()
    words = list(words)
    if prefix is None or prefix == "":
        arr = np.random.choice(words, 1)
    else:
        arr = prefix
    arr = list(arr)
    for j in range(length):
        cnt_zeros = 0
        p_arr = []
        #считаю вероятности для определения следующего слова
        for word in words:
            if arr[-1] + " " + word in two_grams.keys():
                p_arr.append(two_grams[arr[-1] + " " + word] / one_grams[arr[-1]])
            else:
                p_arr.append(0)
                cnt_zeros += 1

        # пару раз была ошибка из-за точности вычислений, поэтому добавил такую проверку
        # x = sum(p_arr)
        # x = max(1 - x, 0)
        # for k in range(len(p_arr)):
        #     if p_arr[k] == 0:
        #         p_arr[k] = x / cnt_zeros

        #рандомно выбираю на основе вероятностей
        arr.append(np.random.choice(words, 1, p=p_arr)[0])
    print(" ".join(arr))


if __name__ == '__main__':
    model = args.model
    with open(model, 'r', encoding='utf-8') as file:
        all_grams = json.load(file)

    length = args.length
    prefix = args.prefix
    make_text(all_grams, length, prefix)
