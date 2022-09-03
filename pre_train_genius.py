import json
import re

import lyricsgenius as genius


def make_data(all_grams):
    with open('artist.json', encoding='utf-8') as f:
        aux = json.load(f)
        lyrics = [song['lyrics'] for song in aux['songs']]
        pos = 0
        for lyric in lyrics:
            lyric = lyric[lyric.find('\n'):]
            lyric = lyric.replace('\n ', '\n')
            lyric = lyric.replace('\n\n', '\n')
            lyric = lyric.replace('Embed', ' ')
            lyric = lyric.replace('Lyrics', ' ')
            lyric = lyric.replace('—', ' ')
            lyric = lyric.replace('»', ' ')
            lyric = lyric.replace('«', ' ')
            lyric = lyric.replace('–', ' ')
            lyric = lyric.replace('…', ' ')
            lyric = lyric.replace('-', ' ')
            lyric = lyric.replace('?', '')
            lyric = lyric.replace(',', '')
            lyric = lyric.replace('.', '')
            lyric = lyric.replace('!', '')
            lyric = lyric.replace(':', '')
            lyric = re.sub("[\(\[].*?[\)\]]", "", lyric)
            lyric = re.sub(r'[^\w\s]+|[\d]+', r'', lyric).strip()
            lyric = re.sub(' +', ' ', lyric)
            lyric = lyric.strip()
            lyric = lyric.lower()
            lyric = lyric.replace('\n', ' ')
            lyric = lyric.replace('\u2005', ' ')
            lyrics[pos] = lyric
            pos += 1
        for lyric in lyrics:
            lyric = re.sub(' +', ' ', lyric)
            arr = lyric.split(' ')
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


def download_dataset(artist, all_grams):  # calling the API
    api = genius.Genius('OtmDtw5-YP0nx3eBs6lXVK23Cn0-gzM_FcodeRol2O-_j58w4JRHd801WF_8lsmS')
    artist = api.search_artist(artist, get_full_info=False)
    artist.save_lyrics(filename='artist', overwrite=True, verbose=True)
    return make_data(all_grams)


def choose_artist():
    with open('all_grams', 'r', encoding='utf-8') as file:
        all_grams = json.load(file)
    artists = ["Слава КПСС", "Pyrokinesis", "Aikko", "Lil krystalll", "oxxxymiron", "ATL", "Og buda", "Loqiemean",
               "163ONMYNECK", "katanacss", "playingtheangel", "booker", "монеточка", "три дня дождя", "MORGENSHTERN",
               "NOIZE MC", "Макс корж", "нексюша", "дора", "boulevard depo", "король и шут", "мэйби бэйби",
               "тима белорусских", "bushido zho", "yanix", "soda luv", "земфира", "лсп", "ANIKV", "элджей",
               "Scally Milano", "КУОК", "The limba", "хаски", "Валентин дядька", "Ежемесячные", 'ssshhhiiittt',
               'билборды', 'электрофорез', 'дайте танк', 'перемотка', 'буерак', 'OST Subway Surfers']
    for artist in artists:
        all_grams = download_dataset(artist, all_grams)
        with open('all_grams.txt', 'w', encoding='utf-8') as file:
            file.write(json.dumps(all_grams))


# скрипт был использован для выкачки текстов песен произвольных артистов указанных в массиве artists  с genius.com
if __name__ == '__main__':
    choose_artist()
