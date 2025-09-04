import json
import time
import wikipedia
import warnings
import logging
from pathlib import Path

# Configure basic service logging
logging.basicConfig(
    level=logging.INFO,
    format="[scraper] %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("scraper")

# Silence the BeautifulSoup parser warning
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')

class WikipediaScraper:
    def __init__(self, data_dir="/app/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def scrape_topic(self, topic):
        """Scrape Wikipedia content for a given topic"""
        logger.info(f"Scraping Wikipedia content for topic: {topic}")
        
        try:
            # Search for relevant pages
            search_results = wikipedia.search(topic, results=5)
            
            scraped_data = {
                "topic": topic,
                "timestamp": time.time(),
                "articles": []
            }
            
            for page_title in search_results:
                try:
                    page = wikipedia.page(page_title)
                    article_data = {
                        "title": page.title,
                        "url": page.url,
                        "content": page.content,
                        "summary": page.summary
                    }
                    scraped_data["articles"].append(article_data)
                    logger.info(f"Scraped: {page.title}")
                    
                except wikipedia.exceptions.DisambiguationError as e:
                    # Try the first option if disambiguation occurs
                    try:
                        page = wikipedia.page(e.options[0])
                        article_data = {
                            "title": page.title,
                            "url": page.url,
                            "content": page.content,
                            "summary": page.summary
                        }
                        scraped_data["articles"].append(article_data)
                        logger.info(f"Scraped (disambiguated): {page.title}")
                    except Exception as inner_e:
                        logger.error(f"Error with disambiguated page: {inner_e}")
                        
                except Exception as e:
                    logger.error(f"Error scraping page {page_title}: {e}")
                    continue
            
            # Save scraped data
            output_file = self.data_dir / f"{topic.replace(' ', '_')}_scraped.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(scraped_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Scraped data saved to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error scraping topic {topic}: {e}")
            return None

def main():
    """Main function to handle scraping requests"""
    scraper = WikipediaScraper()
    
    # Check for topic requests
    request_file = Path("/app/data/scrape_request.txt")
    
    while True:
        if request_file.exists():
            try:
                with open(request_file, 'r') as f:
                    topic = f.read().strip()
                
                if topic:
                    logger.info(f"Processing scrape request for: {topic}")
                    result = scraper.scrape_topic(topic)
                    
                    # Create completion marker
                    if result:
                        completion_file = Path("/app/data/scrape_complete.txt")
                        with open(completion_file, 'w') as f:
                            f.write(str(result))
                    
                    # Remove request file
                    request_file.unlink()
                    
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                request_file.unlink(missing_ok=True)
        
        time.sleep(2)  # Check every 2 seconds

if __name__ == "__main__":
    main()
