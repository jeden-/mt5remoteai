"""
Testy dla modułu Reddit Scraper w systemie NikkeiNinja.
"""
import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from social.reddit_scraper import RedditScraper

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_submission():
    """Tworzy mock obiektu Submission."""
    submission = MagicMock()
    submission.id = "abc123"
    submission.title = "Test Post"
    submission.selftext = "This is a test post"
    submission.score = 100
    submission.upvote_ratio = 0.75
    submission.num_comments = 50
    submission.created_utc = 1677686400  # 2024-03-01
    submission.url = "https://reddit.com/r/test/abc123"
    submission.author = "test_user"
    submission.is_original_content = True
    submission.all_awardings = ["silver", "gold"]
    submission.subreddit = MagicMock()
    submission.subreddit.display_name = "wallstreetbets"
    return submission


@pytest.fixture
def scraper():
    """Tworzy instancję RedditScraper."""
    return RedditScraper()


def test_inicjalizacja(scraper):
    """Test inicjalizacji scrapera."""
    assert scraper.reddit is None
    assert len(scraper.subreddits) == 5
    assert 'wallstreetbets' in scraper.subreddits


def test_inicjalizacja_api(scraper):
    """Test inicjalizacji połączenia z Reddit API."""
    credentials = {
        'client_id': 'test_id',
        'client_secret': 'test_secret',
        'user_agent': 'test_agent'
    }
    
    with patch('praw.Reddit') as mock_reddit:
        scraper.inicjalizuj(credentials)
        mock_reddit.assert_called_once_with(
            client_id='test_id',
            client_secret='test_secret',
            user_agent='test_agent'
        )


def test_analizuj_post(scraper, mock_submission):
    """Test analizy pojedynczego posta."""
    post_data = scraper._analizuj_post(mock_submission)
    
    assert post_data['id'] == "abc123"
    assert post_data['title'] == "Test Post"
    assert post_data['score'] == 100
    assert post_data['upvote_ratio'] == 0.75
    assert post_data['num_comments'] == 50
    assert post_data['author'] == "test_user"
    assert post_data['awards'] == 2
    assert post_data['subreddit'] == "wallstreetbets"


def test_pobierz_posty(scraper, mock_submission):
    """Test pobierania postów."""
    with patch('praw.Reddit') as mock_reddit:
        # Setup
        mock_subreddit = MagicMock()
        mock_subreddit.hot.return_value = [mock_submission]
        mock_reddit.return_value.subreddit.return_value = mock_subreddit
        
        scraper.inicjalizuj({
            'client_id': 'test_id',
            'client_secret': 'test_secret',
            'user_agent': 'test_agent'
        })
        
        # Test
        posty = scraper.pobierz_posty(subreddits=['wallstreetbets'], limit=1)
        
        assert len(posty) == 1
        assert posty[0]['id'] == "abc123"
        assert posty[0]['subreddit'] == "wallstreetbets"


def test_szukaj_posty(scraper, mock_submission):
    """Test wyszukiwania postów."""
    with patch('praw.Reddit') as mock_reddit:
        # Setup
        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = [mock_submission]
        mock_reddit.return_value.subreddit.return_value = mock_subreddit
        
        scraper.inicjalizuj({
            'client_id': 'test_id',
            'client_secret': 'test_secret',
            'user_agent': 'test_agent'
        })
        
        # Test
        posty = scraper.szukaj_posty("test", subreddits=['wallstreetbets'], limit=1)
        
        assert len(posty) == 1
        assert posty[0]['id'] == "abc123"
        assert posty[0]['subreddit'] == "wallstreetbets"


def test_analizuj_sentyment(scraper):
    """Test analizy sentymentu."""
    posty = [
        {
            'id': '1',
            'subreddit': 'wallstreetbets',
            'score': 100,
            'upvote_ratio': 0.8,
            'num_comments': 50
        },
        {
            'id': '2',
            'subreddit': 'wallstreetbets',
            'score': 200,
            'upvote_ratio': 0.9,
            'num_comments': 100
        }
    ]
    
    statystyki = scraper.analizuj_sentyment(posty)
    
    assert statystyki['liczba_postow'] == 2
    assert statystyki['sredni_score'] == 150
    assert abs(statystyki['sredni_upvote_ratio'] - 0.85) < 0.0001
    assert statystyki['srednia_liczba_komentarzy'] == 75
    assert 'wallstreetbets' in statystyki['posty_per_subreddit']
    assert statystyki['posty_per_subreddit']['wallstreetbets']['liczba_postow'] == 2 