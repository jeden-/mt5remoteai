"""
Prosty test połączenia z bazą danych.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import traceback

def test_db_connection():
    # Parametry połączenia jako postgres
    params = {
        'host': 'localhost',
        'port': '5432',
        'database': 'postgres',
        'user': 'postgres',
        'password': 'postgres',
        'client_encoding': 'utf8'
    }

    print("\nPróba połączenia jako postgres...")
    print("Parametry połączenia:")
    for key, value in params.items():
        if key != 'password':
            print(f"- {key}: {value}")
        else:
            print(f"- {key}: *****")

    try:
        print("\nPróba połączenia...")
        conn = psycopg2.connect(**params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        print("Połączenie udane! 🥷")
        
        with conn.cursor() as cur:
            # Reset hasła dla użytkownika ninja
            print("\nResetuję hasło dla użytkownika ninja...")
            cur.execute("ALTER USER ninja WITH PASSWORD 'ninja';")
            print("Hasło zresetowane.")
            
            # Nadanie wszystkich uprawnień do bazy nikkeininja
            print("\nNadaję uprawnienia do bazy nikkeininja...")
            cur.execute("GRANT ALL PRIVILEGES ON DATABASE nikkeininja TO ninja;")
            print("Uprawnienia nadane.")

        conn.close()
        print("\nPołączenie z bazą postgres zamknięte.")
        
        # Połączenie z bazą nikkeininja jako postgres
        print("\nŁączę z bazą nikkeininja jako postgres...")
        params['database'] = 'nikkeininja'
        conn = psycopg2.connect(**params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cur:
            # Nadanie uprawnień do schematu public
            print("\nNadaję uprawnienia do schematu public...")
            cur.execute("GRANT ALL ON SCHEMA public TO ninja;")
            cur.execute("GRANT ALL ON ALL TABLES IN SCHEMA public TO ninja;")
            print("Uprawnienia do schematu nadane.")

        conn.close()
        print("\nPołączenie z bazą nikkeininja zamknięte.")
        
        # Teraz próba połączenia jako ninja
        print("\nPróba połączenia jako ninja...")
        ninja_params = {
            'host': 'localhost',
            'port': '5432',
            'database': 'nikkeininja',
            'user': 'ninja',
            'password': 'ninja',
            'client_encoding': 'utf8'
        }
        
        conn = psycopg2.connect(**ninja_params)
        print("Połączenie jako ninja udane! 🥷")
        
        with conn.cursor() as cur:
            cur.execute('SELECT version()')
            version = cur.fetchone()
            print(f"\nWersja PostgreSQL: {version[0]}")
            
            # Sprawdzamy uprawnienia
            cur.execute("""
                SELECT current_user, session_user, current_database(),
                       has_database_privilege(current_user, current_database(), 'CONNECT'),
                       has_database_privilege(current_user, current_database(), 'CREATE'),
                       has_database_privilege(current_user, current_database(), 'TEMPORARY')
            """)
            session_info = cur.fetchone()
            print("\nInformacje o sesji:")
            print(f"- Aktualny użytkownik: {session_info[0]}")
            print(f"- Użytkownik sesji: {session_info[1]}")
            print(f"- Aktualna baza: {session_info[2]}")
            print("\nUprawnienia:")
            print(f"- CONNECT: {session_info[3]}")
            print(f"- CREATE: {session_info[4]}")
            print(f"- TEMPORARY: {session_info[5]}")

        conn.close()
        print("\nTest zakończony pomyślnie!")

    except psycopg2.Error as e:
        print("\n❌ Błąd połączenia:")
        print(f"Typ błędu: {type(e).__name__}")
        print(f"Treść błędu: {str(e)}")
        if hasattr(e, 'pgcode'):
            print(f"Kod błędu: {e.pgcode}")
        if hasattr(e, 'pgerror'):
            print(f"Szczegóły błędu: {e.pgerror}")
        
        print("\nPełny stack trace:")
        traceback.print_exc()

if __name__ == "__main__":
    test_db_connection() 