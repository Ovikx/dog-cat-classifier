from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from PIL import Image
import io
import aiohttp
import asyncio
from progress_bar import ProgressBar

class ImageDownloader:
    def __init__(self, webdriver_path):
        self.webdriver_path = webdriver_path
        self.options = webdriver.ChromeOptions()
        self.options.add_argument('headless')
        self.driver = webdriver.Chrome(self.webdriver_path, options=self.options)
    
    async def download_task(self, url_data):
        async with aiohttp.ClientSession() as session:
            print('Downloading images...')
            tasks = []
            for url, path, filename in url_data:
                task = asyncio.ensure_future(self.retrieve_image(session, url, path, filename))
                tasks.append(task)

            files = await asyncio.gather(*tasks)
            for image_file, path, filename in files:
                try:
                    if image_file == None:
                        continue
                    image = Image.open(image_file)
                    with open(f'{path}/{filename}', 'wb') as f:
                        try:
                            image.save(f, 'JPEG')
                        except:
                            image.save(f, 'PNG')
                except Exception as e:
                    print(f'DOWNLOAD FAILED for {filename}- {e}')
            print('Images downloaded')
    
    async def retrieve_image(self, session, url, path, filename):
        try:
            async with session.get(url, timeout=5) as response:
                image_file = io.BytesIO(await response.content.read())
                return (image_file, path, filename)
        except:
            print(f'TIMEOUT - {filename}')
            return (None, None, None)

    def download_images(self, query, limit, path):
        self.driver.get('https:images.google.com')
        search_bar = self.driver.find_element_by_name('q')
        search_bar.send_keys(query)
        search_bar.send_keys(Keys.RETURN)
        urls = set()
        pb = ProgressBar(max_value=limit)

        while len(urls) < limit:
            thumbnails = self.driver.find_elements(By.CLASS_NAME, 'Q4LuWd')
            for thumbnail in thumbnails:
                try:
                    thumbnail.click()
                except:
                    pass
                expanded = self.driver.find_elements(By.CLASS_NAME, 'n3VNCb')
                for image in expanded:
                    if image.get_attribute('src') and 'http' in image.get_attribute('src'):
                        old_len = len(urls)
                        urls.add(image.get_attribute('src'))
                        print(f'Search progress: {pb.update(len(urls)-old_len)} {len(urls)}/{limit} done', end='\r')
                if len(urls) >= limit:
                    break

        self.driver.quit()

        url_data = [(url, path, f'{query}{i+1}.jpg') for i, url in enumerate(urls)]
        asyncio.run(self.download_task(url_data))

        return url_data

downloader = ImageDownloader('C:\Program Files (x86)\\chromedriver.exe')
downloader.download_images('cat', 200, 'images/cat')