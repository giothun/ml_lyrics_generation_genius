import json
import os
import re
import argparse
from os import walk

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('-indir', '--input-dir',
                    help='путь к директории, в которой лежит коллекция документов для обучения модели.',
                    default=None)
parser.add_argument('-m', '--model', type=str, help='путь к файлу, в который сохраняется модель.',
                    default="all_grams2.txt")

args = parser.parse_args()


def make_data(texts):
    all_grams = {}
    pos = 0
    for text in texts:
        text = text[text.find('\n'):]
        text = text.replace('\n ', '\n')
        text = text.replace('\n\n', '\n')
        text = text.replace('Embed', ' ')
        text = text.replace('Lyrics', ' ')
        text = text.replace('—', ' ')
        text = text.replace('»', ' ')
        text = text.replace('«', ' ')
        text = text.replace('–', ' ')
        text = text.replace('…', ' ')
        text = text.replace('-', ' ')
        text = text.replace('?', '')
        text = text.replace(',', '')
        text = text.replace('.', '')
        text = text.replace('!', '')
        text = text.replace(':', '')
        text = re.sub("[\(\[].*?[\)\]]", "", text)
        text = re.sub(r'[^\w\s]+|[\d]+', r'', text).strip()
        text = re.sub(' +', ' ', text)
        text = text.strip()
        text = text.lower()
        text = text.replace('\n', ' ')
        text = text.replace('\u2005', ' ')
        texts[pos] = text
        pos += 1
    for text in texts:
        text = re.sub(' +', ' ', text)
        arr = text.split(' ')
        for i in range(len(arr)):
            if i - 2 >= 0:
                tmp = arr[i - 2] + " " + arr[i - 1] + " " + arr[i]
                if tmp in all_grams:
                    all_grams[tmp] += 1
                else:
                    all_grams[tmp] = 1
            if i - 1 >= 0:
                tmp = arr[i - 1] + " " + arr[i]
                if tmp in all_grams:
                    all_grams[tmp] += 1
                else:
                    all_grams[tmp] = 1
            if arr[i] in all_grams:
                all_grams[arr[i]] += 1
            else:
                all_grams[arr[i]] = 1

    return all_grams


if __name__ == '__main__':
    model = args.model
    input_dir = args.input_dir

    texts = []
    if input_dir is None:
        print("Закончите ввод текста строкой end")
        s = ""
        text = ""
        while s != "end":
            s = input()
            if s != "end":
                texts += s + "\n"
        texts.append(text)
    else:
        filenames = next(walk(input_dir), (None, None, []))[2]
        for file in filenames:
            path = os.path.join(input_dir, file)
            text = ""
            with open(path, encoding='utf-8') as file:
                for line in file:
                    text += line.strip() + "\n"
            texts.append(text)
    all_grams = make_data(texts)
    with open(model, 'w', encoding='utf-8') as file:
        file.write(json.dumps(all_grams))
