import json
import re

import lyricsgenius as genius


def make_data():
    with open('artist.json', encoding='utf-8') as f:
        aux = json.load(f)
    lyrics = [song['lyrics'] for song in aux['songs']]
    titles = [song['title'] for song in aux['songs']]
    for i in range(len(titles)):
        lyric = lyrics[i]
        title = titles[i]

        file = aux['name'] + " " + title
        file = re.sub(r'[^\w\s]+|[\d]+', r'', file).strip()
        file = "data/" + file + ".txt"
        with open(file, mode='w+', encoding='utf-8') as f:
            f.write(lyric)


def download_dataset(artist):  # api call
    api = genius.Genius('OtmDtw5-YP0nx3eBs6lXVK23Cn0-gzM_FcodeRol2O-_j58w4JRHd801WF_8lsmS')
    artist = api.search_artist(artist, get_full_info=False)
    artist.save_lyrics(filename='artist', overwrite=True, verbose=True)


def choose_artist():
    artists = ["Слава КПСС", "Pyrokinesis", "Aikko", "Lil krystalll", "oxxxymiron", "ATL", "Og buda", "Loqiemean",
               "163ONMYNECK", "katanacss", "playingtheangel", "booker", "монеточка", "три дня дождя", "MORGENSHTERN",
               "NOIZE MC", "Макс корж", "нексюша", "дора", "boulevard depo", "король и шут", "мэйби бэйби",
               "тима белорусских", "bushido zho", "yanix", "soda luv", "земфира", "лсп", "ANIKV", "элджей",
               "Scally Milano", "КУОК", "The limba", "хаски", "Валентин дядька", "Ежемесячные", 'ssshhhiiittt',
               'билборды', 'электрофорез', 'дайте танк', 'перемотка', 'буерак', 'OST Subway Surfers']

    for artist in artists:
        download_dataset(artist)
        make_data()


# script was used to download lyrics of arbitrary artists specified in the artists array from genius.com
if __name__ == '__main__':
    choose_artist()
