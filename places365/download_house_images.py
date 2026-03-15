from icrawler.builtin import GoogleImageCrawler
import os

classes = ['bedroom interior house', 'kitchen interior house', 'living_room interior house']

for cls in classes:
    class_dir = cls.split()[0].replace('_', '')
    os.makedirs(f'house_data/train/{class_dir}', exist_ok=True)
    crawler = GoogleImageCrawler(storage={'root_dir': f'house_data/train/{class_dir}'})
    print(f"Downloading 2000 images for {cls}...")
    crawler.crawl(keyword=cls, max_num=2000, min_size=(200,200))

print("Download complete - rerun train_house_minimal.py")
