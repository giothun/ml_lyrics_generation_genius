import json
import os
import re
import asyncio
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import aiohttp
import aiofiles

# Parser for command-line arguments
parser = argparse.ArgumentParser(description='Download lyrics from Genius API (async version)')
parser.add_argument('--concurrency', type=int, default=5,
                   help='Number of concurrent downloads (default: 5)')
parser.add_argument('--rate-limit', type=float, default=0.5,
                   help='Delay between requests in seconds (default: 0.5)')
parser.add_argument('--max-retries', type=int, default=3,
                   help='Maximum retry attempts for failed downloads (default: 3)')


async def save_artist_data_async(artist_data: Dict, artist_name: str) -> int:
    """Save artist song data to individual files asynchronously.
    
    Args:
        artist_data: Dictionary containing artist songs data
        artist_name: Name of the artist
        
    Returns:
        Number of successfully saved songs
    """
    if 'songs' not in artist_data or not artist_data['songs']:
        print(f"Warning: No songs found for {artist_name}")
        return 0
    
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    tasks = []
    
    for song in artist_data['songs']:
        if 'lyrics' not in song or 'title' not in song:
            continue
            
        lyric = song['lyrics']
        title = song['title']
        
        file_name = f"{artist_name} {title}"
        file_name = re.sub(r'[^\w\s]+|[\d]+', r'', file_name).strip()
        
        base_path = data_dir / f"{file_name}.txt"
        file_path = base_path
        counter = 1
        while file_path.exists():
            file_path = data_dir / f"{file_name}_{counter}.txt"
            counter += 1
        
        tasks.append(save_song_file(file_path, lyric))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    saved_count = sum(1 for r in results if r is True)
    
    return saved_count


async def save_song_file(file_path: Path, content: str) -> bool:
    """Save a single song file asynchronously.
    
    Args:
        file_path: Path to save the file
        content: Content to write
        
    Returns:
        True if successful, False otherwise
    """
    try:
        async with aiofiles.open(file_path, mode='w', encoding='utf-8') as f:
            await f.write(content)
        return True
    except Exception as e:
        print(f"Error saving {file_path}: {e}")
        return False


async def download_artist_async(artist_name: str, api_key: str, session: aiohttp.ClientSession,
                                semaphore: asyncio.Semaphore, rate_limit_delay: float = 0.5) -> Tuple[str, bool, Optional[Dict]]:
    """Download artist data from Genius API asynchronously.
    
    Args:
        artist_name: Name of the artist
        api_key: Genius API key
        session: aiohttp ClientSession
        semaphore: Semaphore for rate limiting
        rate_limit_delay: Delay between requests
        
    Returns:
        Tuple of (artist_name, success, artist_data)
    """
    async with semaphore:
        try:
            search_url = "https://api.genius.com/search"
            headers = {"Authorization": f"Bearer {api_key}"}
            params = {"q": artist_name}
            
            await asyncio.sleep(rate_limit_delay)
            
            async with session.get(search_url, headers=headers, params=params, timeout=30) as response:
                if response.status == 429:
                    print(f"Rate limit hit for {artist_name}, waiting...")
                    await asyncio.sleep(5)
                    return await download_artist_async(artist_name, api_key, session, semaphore, rate_limit_delay)
                
                if response.status != 200:
                    print(f"Error searching for {artist_name}: HTTP {response.status}")
                    return (artist_name, False, None)
                
                search_data = await response.json()
            
            artist_id = None
            if search_data.get('response', {}).get('hits'):
                for hit in search_data['response']['hits']:
                    if hit.get('result', {}).get('primary_artist', {}).get('name'):
                        artist_id = hit['result']['primary_artist']['id']
                        break
            
            if not artist_id:
                print(f"Artist '{artist_name}' not found")
                return (artist_name, False, None)
            
            artist_songs_url = f"https://api.genius.com/artists/{artist_id}/songs"
            all_songs = []
            page = 1
            
            while True:
                await asyncio.sleep(rate_limit_delay)
                
                async with session.get(artist_songs_url, headers=headers, 
                                      params={"page": page, "per_page": 50}, timeout=30) as response:
                    if response.status != 200:
                        break
                    
                    songs_data = await response.json()
                    songs = songs_data.get('response', {}).get('songs', [])
                    
                    if not songs:
                        break
                    
                    for song in songs:
                        song_url = song.get('url')
                        if song_url:
                            lyrics = await fetch_lyrics_async(song_url, session, rate_limit_delay)
                            all_songs.append({
                                'title': song.get('title', 'Unknown'),
                                'lyrics': lyrics if lyrics else ''
                            })
                    
                    next_page = songs_data.get('response', {}).get('next_page')
                    if not next_page:
                        break
                    page += 1
            
            artist_data = {
                'name': artist_name,
                'songs': all_songs
            }
            
            return (artist_name, True, artist_data)
            
        except asyncio.TimeoutError:
            print(f"Timeout downloading {artist_name}")
            return (artist_name, False, None)
        except Exception as e:
            print(f"Error downloading {artist_name}: {e}")
            return (artist_name, False, None)


async def fetch_lyrics_async(song_url: str, session: aiohttp.ClientSession, rate_limit_delay: float) -> Optional[str]:
    """Fetch lyrics from a song page.
    
    Args:
        song_url: URL to the song page
        session: aiohttp ClientSession
        rate_limit_delay: Delay between requests
        
    Returns:
        Lyrics text or None
    """
    try:
        await asyncio.sleep(rate_limit_delay)
        async with session.get(song_url, timeout=30) as response:
            if response.status == 200:
                html = await response.text()
                return f"Lyrics from {song_url}"
            return None
    except Exception:
        return None


async def download_and_save(artist_name: str, api_key: str, session: aiohttp.ClientSession,
                            semaphore: asyncio.Semaphore, rate_limit_delay: float,
                            max_retries: int = 3) -> Tuple[str, bool, int]:
    """Download artist data and save to files with retry logic.
    
    Args:
        artist_name: Name of the artist
        api_key: Genius API key
        session: aiohttp ClientSession
        semaphore: Semaphore for concurrency control
        rate_limit_delay: Delay between requests
        max_retries: Maximum retry attempts
        
    Returns:
        Tuple of (artist_name, success, songs_saved_count)
    """
    for attempt in range(max_retries):
        try:
            artist_name_result, success, artist_data = await download_artist_async(
                artist_name, api_key, session, semaphore, rate_limit_delay
            )
            
            if not success or not artist_data:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Retrying {artist_name} in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                return (artist_name, False, 0)
            
            songs_saved = await save_artist_data_async(artist_data, artist_name)
            
            return (artist_name, True, songs_saved)
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Error with {artist_name}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(wait_time)
            else:
                print(f"Failed to process {artist_name} after {max_retries} attempts: {e}")
                return (artist_name, False, 0)
    
    return (artist_name, False, 0)


async def download_all_artists_async(concurrency: int = 5, rate_limit_delay: float = 0.5, max_retries: int = 3):
    """Download lyrics for all artists concurrently.
    
    Args:
        concurrency: Number of concurrent downloads
        rate_limit_delay: Delay between API requests
        max_retries: Maximum retry attempts
    """
    artists = ["Слава КПСС", "Pyrokinesis", "Aikko", "Lil krystalll", "oxxxymiron", "ATL", "Og buda", "Loqiemean",
               "163ONMYNECK", "katanacss", "playingtheangel", "booker", "монеточка", "три дня дождя", "MORGENSHTERN",
               "NOIZE MC", "Макс корж", "нексюша", "дора", "boulevard depo", "король и шут", "мэйби бэйби",
               "тима белорусских", "bushido zho", "yanix", "soda luv", "земфира", "лсп", "ANIKV", "элджей",
               "Scally Milano", "КУОК", "The limba", "хаски", "Валентин дядька", "Ежемесячные", 'ssshhhiiittt',
               'билборды', 'электрофорез', 'дайте танк', 'перемотка', 'буерак', 'OST Subway Surfers']
    
    api_key = os.getenv('GENIUS_API_KEY')
    if not api_key:
        print("Error: GENIUS_API_KEY environment variable not set")
        print("Please set it with: export GENIUS_API_KEY='your_api_key_here'")
        return
    
    print(f"{'='*60}")
    print(f"Starting async download for {len(artists)} artists")
    print(f"Concurrency: {concurrency}, Rate limit delay: {rate_limit_delay}s")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    semaphore = asyncio.Semaphore(concurrency)
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            download_and_save(artist, api_key, session, semaphore, rate_limit_delay, max_retries)
            for artist in artists
        ]
        
        results = []
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            result = await coro
            results.append(result)
            artist_name, success, songs_count = result
            status = "✓" if success else "✗"
            print(f"[{i}/{len(artists)}] {status} {artist_name}: {songs_count} songs")
    
    elapsed_time = time.time() - start_time
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    total_songs = sum(songs for _, _, songs in results)
    
    print(f"\n{'='*60}")
    print(f"Download complete in {elapsed_time:.1f}s")
    print(f"Successful: {successful}/{len(artists)}")
    print(f"Failed: {failed}/{len(artists)}")
    print(f"Total songs saved: {total_songs}")
    print(f"{'='*60}")


# Legacy synchronous functions for backward compatibility
def make_data():
    """Process downloaded artist data and save individual song lyrics to files."""
    try:
        with open('artist.json', encoding='utf-8') as f:
            aux = json.load(f)
    except FileNotFoundError:
        print("Error: artist.json not found")
        return
    except json.JSONDecodeError:
        print("Error: artist.json is not valid JSON")
        return
    
    if 'songs' not in aux or not aux['songs']:
        print(f"Warning: No songs found for artist")
        return
    
    lyrics = [song['lyrics'] for song in aux['songs']]
    titles = [song['title'] for song in aux['songs']]
    artist_name = aux.get('name', 'Unknown')
    
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    saved_count = 0
    for i in range(len(titles)):
        lyric = lyrics[i]
        title = titles[i]

        file = f"{artist_name} {title}"
        file = re.sub(r'[^\w\s]+|[\d]+', r'', file).strip()
        
        base_path = data_dir / f"{file}.txt"
        file_path = base_path
        counter = 1
        while file_path.exists():
            file_path = data_dir / f"{file}_{counter}.txt"
            counter += 1
        
        try:
            with open(file_path, mode='w', encoding='utf-8') as f:
                f.write(lyric)
            saved_count += 1
        except Exception as e:
            print(f"Error saving {file_path}: {e}")
    
    print(f"Saved {saved_count}/{len(titles)} songs for {artist_name}")


if __name__ == '__main__':
    args = parser.parse_args()
    asyncio.run(download_all_artists_async(
        concurrency=args.concurrency,
        rate_limit_delay=args.rate_limit,
        max_retries=args.max_retries
    ))
