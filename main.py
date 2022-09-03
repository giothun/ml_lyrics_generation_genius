import json
import pickle
import numpy as np


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


def make_text(all_grams, words, length, prefix=None):
    one_grams, two_grams, three_grams = separate_grams(all_grams)
    if prefix is None or prefix == "":
        arr = np.random.choice(words, 1)
    else:
        arr = prefix.split(' ')
    arr = list(arr)
    for j in range(length):
        cnt_zeros = 0
        p_arr = []
        for word in words:
            if arr[-1] + " " + word in two_grams.keys():
                p_arr.append(two_grams[arr[-1] + " " + word] / one_grams[arr[-1]])
            else:
                p_arr.append(0)
                cnt_zeros += 1
        x = sum(p_arr)
        x = max(1 - x, 0)
        for k in range(len(p_arr)):
            if p_arr[k] == 0:
                p_arr[k] = x / cnt_zeros

        arr.append(np.random.choice(words, 1, p=p_arr)[0])
    print(" ".join(arr))


if __name__ == '__main__':
    with open('single.txt', 'r', encoding='utf-8') as file:
        all_grams = json.load(file)
    with open("words.txt", "rb") as file:
        words = pickle.load(file)

    length = int(input("Введите длину генерируемой последовательности: "))
    prefix = input("Введите префикс (префикс может быть пустой строкой): ")
    make_text(all_grams, words, length, prefix)
