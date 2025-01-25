"""
Testy dla monitora strategii.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from monitoring.monitor_strategii import MonitorStrategii
from strategie.interfejs import SygnalTransakcyjny, KierunekTransakcji
from baza_danych.baza import BazaDanych


@pytest.fixture
def mock_baza():
    """Tworzy mock bazy danych."""
    return Mock(spec=BazaDanych)


@pytest.fixture
def monitor(mock_baza):
    """Tworzy instancję monitora z mockiem bazy danych."""
    return MonitorStrategii(mock_baza, okno_czasu=timedelta(hours=24))


@pytest.fixture
def przykladowy_sygnal():
    """Tworzy przykładowy sygnał transakcyjny."""
    return SygnalTransakcyjny(
        timestamp=datetime.now(),
        symbol="NKY",
        kierunek=KierunekTransakcji.LONG,
        cena_wejscia=100.0,
        stop_loss=None,
        take_profit=None,
        wolumen=1.0,
        opis="Test",
        metadane={
            'zysk_procent': 2.5,
            'sentyment': 0.8,
            'sentyment_pewnosc': 0.7,
            'waga': 0.6
        }
    )


def test_inicjalizacja(monitor):
    """Test inicjalizacji monitora."""
    assert monitor.okno_czasu == timedelta(hours=24)
    assert len(monitor.historia) == 0
    assert monitor.statystyki_cache == {}


def test_dodaj_sygnal(monitor, przykladowy_sygnal, mock_baza):
    """Test dodawania sygnału."""
    monitor.dodaj_sygnal(przykladowy_sygnal)
    
    assert len(monitor.historia) == 1
    assert monitor.historia[0] == przykladowy_sygnal
    mock_baza.dodaj_metryki.assert_called_once()


def test_aktualizuj_statystyki(monitor, przykladowy_sygnal):
    """Test aktualizacji statystyk."""
    # Dodaj kilka sygnałów
    for _ in range(3):
        monitor.dodaj_sygnal(przykladowy_sygnal)
    
    stats = monitor.pobierz_statystyki()
    
    assert stats['liczba_transakcji'] == 3
    assert stats['win_rate'] == 1.0  # Wszystkie zyskowne
    assert stats['liczba_long'] == 3
    assert stats['liczba_short'] == 0
    assert stats['sredni_sentyment'] == 0.8
    assert stats['srednia_waga'] == 0.6


def test_generuj_raport(monitor, przykladowy_sygnal):
    """Test generowania raportu."""
    monitor.dodaj_sygnal(przykladowy_sygnal)
    raport = monitor.generuj_raport()
    
    assert "📊 Raport strategii" in raport
    assert "Liczba transakcji: 1" in raport
    assert "Win rate: 100.00%" in raport
    assert "Long: 1" in raport
    assert "Średni sentyment: 0.80" in raport


def test_brak_danych(monitor):
    """Test zachowania przy braku danych."""
    stats = monitor.pobierz_statystyki()
    assert stats == {}
    
    raport = monitor.generuj_raport()
    assert raport == "Brak danych do wygenerowania raportu"


def test_aktualizacja_cache(monitor, przykladowy_sygnal):
    """Test aktualizacji cache statystyk."""
    monitor.dodaj_sygnal(przykladowy_sygnal)
    pierwsze_stats = monitor.pobierz_statystyki()
    
    # Symuluj upływ czasu
    monitor.ostatnia_aktualizacja -= timedelta(minutes=10)
    
    drugie_stats = monitor.pobierz_statystyki()
    assert drugie_stats['timestamp'] > pierwsze_stats['timestamp']


def test_filtrowanie_historii(monitor, przykladowy_sygnal):
    """Test filtrowania historii po oknie czasowym."""
    # Dodaj stary sygnał
    stary_sygnal = przykladowy_sygnal
    stary_sygnal.timestamp = datetime.now() - timedelta(hours=25)
    monitor.dodaj_sygnal(stary_sygnal)
    
    # Dodaj nowy sygnał
    nowy_sygnal = przykladowy_sygnal
    nowy_sygnal.timestamp = datetime.now()
    monitor.dodaj_sygnal(nowy_sygnal)
    
    stats = monitor.pobierz_statystyki()
    assert stats['liczba_transakcji'] == 1  # Tylko nowy sygnał 