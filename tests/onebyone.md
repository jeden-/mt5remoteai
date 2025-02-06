# Wyniki testów pojedynczych plików 🧪

1. test_technical_indicators.py ✅

**Status**: Wszystkie testy zaliczone
**Liczba testów**: 31
**Czas wykonania**: 2.53s

**Pokrycie kodu**:
- Całkowita liczba linii: 2560
- Niepokryte linie: 2007
- Pokrycie całkowite: 21.60%
- Pokrycie src/utils/technical_indicators.py: 84%

**Brakujące pokrycie w technical_indicators.py**:
- Linie 50-57: Walidacja parametrów
- Linie 501-519: Wykrywanie wzorców
- Linie 534-548: Wykrywanie dywergencji
- Linie 566-579: Funkcje pomocnicze

**Lista testów**:
1. test_init_default_params ✅
2. test_init_custom_params ✅
3. test_calculate_all_invalid_input ✅
4. test_calculate_all_missing_columns ✅
5. test_calculate_all_empty_dataframe ✅
6. test_moving_averages ✅
7. test_rsi ✅
8. test_macd ✅
9. test_stochastic ✅
10. test_atr ✅
11. test_momentum ✅
12. test_support_resistance ✅
13. test_detect_patterns ✅
14. test_calculate_sma ✅
15. test_calculate_ema ✅
16. test_calculate_rsi ✅
17. test_calculate_macd ✅
18. test_calculate_bollinger_bands ✅
19. test_calculate_stochastic ✅
20. test_calculate_atr ✅
21. test_calculate_adx ✅
22. test_detect_patterns_detailed ✅
23. test_detect_patterns_validation ✅
24. test_detect_patterns_edge_cases ✅
25. test_input_validation_edge_cases ✅
26. test_pivot_points_detailed ✅
27. test_divergence_detection ✅
28. test_helper_functions_edge_cases ✅

2. test_context_provider.py ✅

**Status**: Wszystkie testy zaliczone
**Liczba testów**: 16
**Czas wykonania**: 19.54s

**Pokrycie kodu**:
- Całkowita liczba linii: 2560
- Niepokryte linie: 2132
- Pokrycie całkowite: 16.72%
- Pokrycie src/rag/context_provider.py: 97%

**Brakujące pokrycie w context_provider.py**:
- Linia 53: Walidacja parametrów
- Linia 78: Obsługa błędów

**Lista testów**:
1. test_initialization ✅
2. test_get_market_context ✅
3. test_format_market_data ✅
4. test_error_handling ✅
5. test_context_limits ✅
6. test_generate_context_summary ✅
7. test_input_validation ✅
8. test_format_market_data_with_many_indicators ✅
9. test_timeframe_formats ✅
10. test_empty_indicators ✅
11. test_timestamp_validation ✅
12. test_memory_usage ✅
13. test_get_market_context_performance ✅
14. test_format_market_data_performance ✅
15. test_generate_summary_performance ✅
16. test_memory_usage ✅

**Wyniki wydajności**:
1. test_generate_summary_performance:
   - Średni czas: 6.58 μs
   - OPS: 151,876 op/s
2. test_format_market_data_performance:
   - Średni czas: 10.42 μs
   - OPS: 96,002 op/s
3. test_get_market_context_performance:
   - Średni czas: 82.80 μs
   - OPS: 12,077 op/s

3. test_logger.py ✅

**Status**: Wszystkie testy zaliczone
**Liczba testów**: 42
**Czas wykonania**: 2.77s

**Pokrycie kodu**:

- Całkowita liczba linii: 2560
- Niepokryte linie: 2056
- Pokrycie całkowite: 19.69%
- Pokrycie src/utils/logger.py: 96%

**Brakujące pokrycie w logger.py**:
- Linie 268-270: Archiwizacja logów
- Linie 345-346: Obsługa błędów JSON
- Linie 410-413: Walidacja ścieżek

**Lista testów**:
1. test_logger_initialization ✅
2. test_log_trade ✅
3. test_log_error ✅
4. test_log_warning ✅
5. test_log_ai_analysis ✅
6. test_get_logs_path ✅
7. test_clear_logs ✅
8. test_archive_logs ✅
9. test_log_strategy_stats ✅
10. test_log_market_data ✅
11. test_log_performance ✅
12. test_log_trade_to_json_success ✅
13. test_log_trade_to_json_error ✅
14. test_get_log_file_path ✅
15. test_get_json_file_path ✅
16. test_get_archive_dir_path ✅
17. test_log_trade_error ✅
18. test_log_ai_analysis_error ✅
19. test_archive_logs_file_not_found ✅
20. test_archive_logs_permission_error ✅
21. test_archive_logs_io_error ✅
22. test_log_signal ✅
23. test_log_trade_to_json_file_not_exists ✅
24. test_log_trade_json_write_error ✅
25. test_log_trade_to_json_write_error ✅
26. test_archive_logs_copy_error ✅
27. test_archive_logs_create_archive_dir_error ✅
28. test_get_logger ✅
29. test_basic_logging_methods ✅
30. test_private_path_methods ✅
31. test_logger_levels[DEBUG] ✅
32. test_logger_levels[INFO] ✅
33. test_logger_levels[WARNING] ✅
34. test_logger_levels[ERROR] ✅
35. test_logger_levels[CRITICAL] ✅
36. test_log_trade_to_json_permission_error ✅
37. test_log_trade_to_json_type_error ✅
38. test_log_trade_with_missing_fields ✅
39. test_log_ai_analysis_with_missing_fields ✅
40. test_log_trade_with_invalid_json ✅
41. test_archive_logs_with_existing_archive ✅
42. test_log_trade_to_json_with_existing_data ✅

4. test_config.py ❌

**Status**: 21 testów zaliczonych, 1 test niezaliczony
**Liczba testów**: 22
**Czas wykonania**: 2.34s


**Pokrycie kodu**:
- Całkowita liczba linii: 2560
- Niepokryte linie: 2118
- Pokrycie całkowite: 17.27%
- Pokrycie src/utils/config.py: 97%

**Brakujące pokrycie w config.py**:
- Linia 145, 147: Obsługa błędów
- Linia 168, 170: Walidacja konfiguracji
- Linia 255: Operacje na plikach

**Lista testów**:
1. test_config_initialization ✅
2. test_config_initialization_with_kwargs ✅
3. test_config_validation_mt5_credentials ✅
4. test_config_validation_db_credentials ✅
5. test_config_validation_trading_params ✅
6. test_config_validation_strategy_params ✅
7. test_config_validation_log_level ✅
8. test_config_loader_load_config ✅
9. test_config_loader_validate_trading_config ✅
10. test_config_loader_validate_strategy_config ✅
11. test_config_loader_validate_database_config ✅
12. test_config_loader_validate_logging_config ✅
13. test_config_loader_merge_configs ✅
14. test_config_loader_load_env_variables ✅
15. test_config_loader_parse_error ✅
16. test_config_loader_save_config ✅
17. test_config_loader_validate_dependencies ✅
18. test_config_loader_parse_error_details ✅
19. test_config_loader_save_error ✅
20. test_config_loader_get_default_config ✅
21. test_config_loader_file_operations ✅
22. test_config_loader_load_error ❌ (FileNotFoundError: Nie znaleziono pliku konfiguracyjnego)

5. test_position_manager.py ⚠️

**Status**: 67 testów zaliczonych, 1 test pominięty
**Liczba testów**: 68
**Czas wykonania**: 5.58s

**Pokrycie kodu**:

- Całkowita liczba linii: 2560
- Niepokryte linie: 2044
- Pokrycie całkowite: 20.16%
- Pokrycie src/trading/position_manager.py: 91%

**Brakujące pokrycie w position_manager.py**:
- Linie 94-99: Walidacja pozycji
- Linia 119: Obsługa błędów otwarcia
- Linie 250-253: Aktualizacja trailing stop
- Linie 263, 265: Obsługa błędów zamknięcia
- Linie 274-276: Walidacja wielkości pozycji
- Linie 280-281: Obliczanie ryzyka
- Linie 379, 457: Obsługa timeoutów

**Lista testów**:
1. test_initialization ✅
2. test_open_position ✅
3. test_open_position_max_size_exceeded ✅
4. test_close_position ✅
5. test_close_position_with_profit ✅
6. test_close_position_with_loss ✅
7. test_check_stop_loss ✅
8. test_check_take_profit ✅
9. test_process_price_update_stop_loss ✅
10. test_process_price_update_take_profit ✅
11. test_process_price_update_no_action ✅
12. test_calculate_position_profit ✅
13. test_calculate_position_pips ✅
14. test_validate_position_size ✅
15. test_get_position_summary ✅
16. test_multiple_positions ✅
17. test_position_risk_reward_ratio ✅
18. test_position_exposure ✅
19. test_check_stop_loss_sell ✅
20. test_check_take_profit_sell ✅
[... i 47 innych testów zaliczonych]
68. test_position_lifecycle ⚠️ (pominięty - brak obsługi async)

**Wyniki wydajności**:
- test_position_calculations_speed:
  - Średni czas: 9.71 μs
  - OPS: 102,936 op/s

**Ostrzeżenia**:
1. Nieznany marker pytest.mark.memory
2. Przestarzałe użycie event_loop fixture
3. Brak obsługi async def dla test_position_lifecycle

6. test_data_models.py

✅ Wszystkie 54 testy przeszły pomyślnie

## Pokrycie kodu:
- Całkowita liczba linii: 2560
- Niepokryte linie: 2159  

- Procent pokrycia: 15.66%

## Szczegóły pokrycia dla kluczowych modułów:
- src/models/data_models.py: 88% pokrycia
  - Brakujące pokrycie w liniach: 128, 183, 198, 214, 271, 280, 316, 321, 326, 331-333, 338-348, 353-363
- src/models/enums.py: 87% pokrycia
  - Brakujące pokrycie w liniach: 14, 27, 44, 56, 66, 79
- src/models/validators.py: 78% pokrycia
  - Brakujące pokrycie w liniach: 31, 49, 125-127, 180, 199-200, 213-214, 227-228, 241-242, 256-257

## Moduły z zerowym pokryciem:
- src/backtest/*
- src/connectors/* 
- src/trading/*
- src/strategies/*
- src/rag/*
- src/utils/technical_indicators.py
- src/utils/prompts.py

7. test_trade_type.py

✅ Wszystkie 7 testów przeszło pomyślnie

## Pokrycie kodu:
- Całkowita liczba linii: 2560
- Niepokryte linie: 2243

- Procent pokrycia: 12.38%

## Szczegóły pokrycia dla kluczowych modułów:
- src/trading/trade_type.py: 100% pokrycia
- src/models/data_models.py: 60% pokrycia
- src/models/enums.py: 87% pokrycia
- src/models/validators.py: 49% pokrycia

## Lista testów:
1. test_trade_type_values ✅
2. test_trade_type_comparison ✅
3. test_trade_type_from_string ✅
4. test_trade_type_invalid_value ✅
5. test_trade_type_str_representation ✅
6. test_trade_type_name_property ✅
7. test_trade_type_opposite ✅

## Moduły z zerowym pokryciem:
- src/backtest/*
- src/connectors/*
- src/trading/* (oprócz trade_type.py)
- src/strategies/*
- src/rag/*
- src/utils/technical_indicators.py
- src/utils/prompts.py

# test_position_status.py

✅ Wszystkie 6 testów przeszło pomyślnie

## Pokrycie kodu:
- Całkowita liczba linii: 2560
- Niepokryte linie: 2240
- Procent pokrycia: 12.50%

## Szczegóły pokrycia dla kluczowych modułów:
- src/trading/position_status.py: 100% pokrycia
- src/models/data_models.py: 60% pokrycia
- src/models/enums.py: 87% pokrycia
- src/models/validators.py: 49% pokrycia

## Lista testów:
1. test_position_status_values ✅
2. test_position_status_comparison ✅
3. test_position_status_from_string ✅
4. test_position_status_invalid_value ✅
5. test_position_status_str_representation ✅
6. test_position_status_name_property ✅

## Moduły z zerowym pokryciem:
- src/backtest/*
- src/connectors/*
- src/trading/* (oprócz position_status.py)
- src/strategies/*
- src/rag/*
- src/utils/technical_indicators.py
- src/utils/prompts.py

# test_basic_strategy.py

❌ 6 testów nie przeszło, 8 testów przeszło pomyślnie

## Pokrycie kodu:
- Całkowita liczba linii: 2560
- Niepokryte linie: 2016
- Procent pokrycia: 21.25%

## Szczegóły pokrycia dla kluczowych modułów:
- src/strategies/basic_strategy.py: 71% pokrycia
  - Brakujące pokrycie w liniach: 35-53, 83, 91, 102-106, 133, 137-138, 164-168, 201-203, 227-229, 253-255, 331-333, 347, 400-402, 418-419, 425-426, 432-433, 439-440, 444-445, 448-449, 452-453, 456-457, 461-463
- src/strategies/base_strategy.py: 57% pokrycia
  - Brakujące pokrycie w liniach: 54, 67, 80, 92-109

## Lista testów:
1. test_strategy_initialization ✅
2. test_generate_signals_no_position ❌
3. test_generate_signals_with_position ❌
4. test_generate_signals_error_handling ❌
5. test_analyze_market ❌
6. test_execute_signals ✅
7. test_calculate_position_size ✅
8. test_calculate_stop_loss ✅
9. test_calculate_take_profit ✅
10. test_strategy_performance_metrics ✅
11. test_strategy_analysis_speed ✅
12. test_strategy_memory_usage ✅
13. test_strategy_recovery ❌
14. test_strategy_integration ❌

## Błędy:
1. AttributeError: 'BasicStrategy' object has no attribute 'logger'
2. AttributeError: 'BasicStrategy' object has no attribute 'mt5_connector'

## Wyniki wydajności:
- test_strategy_analysis_speed:
  - Średni czas: 9.70 ms
  - OPS: 103.12 op/s

## Ostrzeżenia:
1. Nieznany marker pytest.mark.memory
2. Przestarzałe użycie event_loop fixture

## Moduły z zerowym pokryciem:
- src/backtest/*
- src/connectors/* (oprócz interfaces/connectors.py)
- src/trading/* (oprócz position_status.py)
- src/rag/*
- src/utils/prompts.py

# test_mt5_handler.py

✅ 24 testy przeszły pomyślnie, 1 test pominięty

## Pokrycie kodu:
- Całkowita liczba linii: 2560
- Niepokryte linie: 2081
- Procent pokrycia: 18.71%

## Szczegóły pokrycia dla kluczowych modułów:
- src/trading/mt5_handler.py: 93% pokrycia
  - Brakujące pokrycie w liniach: 85, 128, 130, 269-277, 375
- src/trading/trade_type.py: 100% pokrycia
- src/trading/position_status.py: 100% pokrycia

## Lista testów:
1. test_initialization ✅
2. test_initialization_error ✅
3. test_open_position ✅
4. test_open_position_error ✅
5. test_close_position ✅
6. test_close_position_no_position ✅
7. test_get_current_price ✅
8. test_get_historical_data ✅
9. test_get_historical_data_error ✅
10. test_get_account_info ✅
11. test_get_positions ✅
12. test_convert_timeframe ✅
13. test_cleanup ✅
14. test_invalid_symbol ✅
15. test_invalid_timeframe ✅
16. test_invalid_volume ✅
17. test_invalid_sl_tp ✅
18. test_invalid_direction ✅
19. test_symbol_min_volume ✅
20. test_symbol_max_volume ✅
21. test_concurrent_operations ✅
22. test_strategy_execution_speed ✅
23. test_memory_usage ✅
24. test_full_trading_cycle ⚠️ (pominięty - brak obsługi async)
25. test_error_handling_under_load ✅

## Wyniki wydajności:
- test_strategy_execution_speed:
  - Średni czas: 2.64 μs
  - OPS: 379.31 kop/s

## Ostrzeżenia:
1. Nieznany marker pytest.mark.memory
2. Przestarzałe użycie event_loop fixture
3. Coroutine 'MT5Handler.get_current_price' was never awaited
4. Brak obsługi async def dla test_full_trading_cycle

## Moduły z zerowym pokryciem:
- src/backtest/*
- src/connectors/*
- src/trading/position_manager.py
- src/rag/*
- src/utils/technical_indicators.py
- src/utils/prompts.py

# test_mt5_connector.py

✅ Wszystkie 10 testów przeszło pomyślnie

## Pokrycie kodu:
- Całkowita liczba linii: 2560
- Niepokryte linie: 2184
- Procent pokrycia: 14.69%

## Szczegóły pokrycia dla kluczowych modułów:
- src/connectors/mt5_connector.py: 100% pokrycia
- src/models/data_models.py: 60% pokrycia
- src/models/enums.py: 87% pokrycia
- src/models/validators.py: 49% pokrycia

## Lista testów:
1. test_connect_success ✅
2. test_connect_initialize_failure ✅
3. test_connect_login_failure ✅
4. test_disconnect ✅
5. test_get_account_info_success ✅
6. test_get_account_info_not_connected ✅
7. test_get_account_info_failure ✅
8. test_get_symbols_success ✅
9. test_get_symbols_not_connected ✅
10. test_get_symbols_failure ✅

## Ostrzeżenia:
1. Przestarzałe użycie event_loop fixture

## Moduły z zerowym pokryciem:
- src/backtest/*
- src/connectors/* (oprócz mt5_connector.py)
- src/trading/*
- src/rag/*
- src/utils/technical_indicators.py
- src/utils/prompts.py

# test_db_integration.py

✅ Wszystkie 8 testów przeszło pomyślnie

## Pokrycie kodu:
- Całkowita liczba linii: 2560
- Niepokryte linie: 2197
- Procent pokrycia: 14.18%

## Szczegóły pokrycia dla kluczowych modułów:
- src/database/postgres_handler.py: 73% pokrycia
  - Brakujące pokrycie w liniach: 39, 92-96, 113-115, 134, 153, 171, 180, 253-285, 296-301
- src/models/data_models.py: 60% pokrycia
- src/models/enums.py: 87% pokrycia
- src/models/validators.py: 49% pokrycia

## Lista testów:
1. test_db_connection ✅
2. test_create_tables ✅
3. test_market_data_operations ✅
4. test_trade_operations ✅
5. test_historical_data_operations ✅
6. test_concurrent_operations ✅
7. test_error_handling ✅
8. test_transaction_rollback ✅

## Ostrzeżenia:
1. Przestarzałe użycie event_loop fixture

## Moduły z zerowym pokryciem:
- src/backtest/*
- src/connectors/*
- src/trading/*
- src/rag/*
- src/utils/technical_indicators.py
- src/utils/prompts.py

# test_market_memory.py

✅ Wszystkie 49 testów przeszło pomyślnie

## Pokrycie kodu:
- Całkowita liczba linii: 2560
- Niepokryte linie: 2061
- Procent pokrycia: 19.49%

## Szczegóły pokrycia dla kluczowych modułów:
- src/rag/market_memory.py: 94% pokrycia
  - Brakujące pokrycie w liniach: 82, 275, 286, 316-318, 341-342, 355-357
- src/models/data_models.py: 60% pokrycia
- src/models/enums.py: 87% pokrycia
- src/models/validators.py: 49% pokrycia

## Lista testów:
### TestPodstawoweFunkcje:
1. test_inicjalizacja_market_memory ✅
2. test_dodawanie_danych_rynkowych ✅

### TestWalidacja:
3. test_walidacja_danych_wejsciowych ✅
4. test_walidacja_wyszukiwania ✅
5. test_walidacja_usuwania ✅

### TestOperacjeDanych:
6. test_wyszukiwanie_danych ✅
7. test_usuwanie_danych ✅

### TestWydajnosc:
8. test_duze_dane ✅
9. test_wiele_symboli_i_timeframes ✅
10. test_operacje_wspolbiezne ✅
11. test_zuzycie_pamieci ✅

### TestObslugaBledow:
12-49. [38 testów obsługi błędów] ✅

## Czas wykonania: 73.77s

## Moduły z zerowym pokryciem:
- src/backtest/*
- src/connectors/*
- src/trading/*
- src/rag/* (oprócz market_memory.py)
- src/utils/technical_indicators.py
- src/utils/prompts.py

# test_data_loader.py

✅ Wszystkie 39 testów przeszło pomyślnie

## Pokrycie kodu:
- Całkowita liczba linii: 2560
- Niepokryte linie: 1969
- Procent pokrycia: 23.09%

## Szczegóły pokrycia dla kluczowych modułów:
- src/backtest/data_loader.py: 94% pokrycia
  - Brakujące pokrycie w liniach: 78-79, 144-146, 332-337
- src/backtest/backtester.py: 17% pokrycia
- src/backtest/performance_metrics.py: 38% pokrycia
- src/backtest/visualizer.py: 22% pokrycia

## Lista testów:
1. test_initialization ✅
2. test_initialization_with_db ✅
3. test_load_from_database_success ✅
4. test_load_from_database_empty_result ✅
5. test_load_from_database_db_error ✅
6. test_load_from_database_invalid_data ✅
7. test_load_from_database_missing_columns ✅
8. test_add_indicators ✅
9. test_add_indicators_empty_data ✅
10. test_add_indicators_missing_columns ✅
11. test_add_indicators_nan_values ✅
12. test_add_indicators_single_row ✅
13. test_add_indicators_invalid_values ✅
14. test_add_indicators_extreme_values ✅
15. test_add_indicators_zero_values ✅
16. test_load_from_mt5_success ✅
17. test_load_from_mt5_initialization_error ✅
18. test_load_from_mt5_copy_rates_error ✅
19. test_load_from_mt5_invalid_timeframe ✅
20. test_load_from_mt5_empty_data ✅
21. test_load_from_mt5_invalid_data ✅
22. test_load_from_mt5_missing_columns ✅
23. test_load_data_from_database_first ✅
24. test_load_data_fallback_to_mt5 ✅
25. test_add_indicators_calculation_error ✅
26. test_load_data_no_db_handler ✅
27. test_load_data_db_empty_result ✅
28. test_timeframe_validation ✅
29. test_start_date_default ✅
30. test_add_indicators_bollinger_validation ✅
31. test_add_indicators_macd_validation ✅
32. test_add_indicators_rsi_validation ✅
33. test_add_indicators_all_nan ✅
34. test_save_to_database_success ✅
35. test_save_to_database_empty_data ✅
36. test_save_to_database_missing_columns ✅
37. test_save_to_database_invalid_data ✅
38. test_save_to_database_db_error ✅
39. test_save_to_database_nan_values ✅

## Ostrzeżenia:
1. Przestarzałe użycie event_loop fixture

## Moduły z zerowym pokryciem:
- src/connectors/*
- src/trading/*
- src/rag/*
- src/utils/prompts.py


# test_context_provider.py ✅

**Status**: Wszystkie testy zaliczone
**Liczba testów**: 16
**Czas wykonania**: 19.37s

**Pokrycie kodu**:
- Całkowita liczba linii: 2560
- Niepokryte linie: 2132
- Pokrycie całkowite: 16.72%
- Pokrycie src/rag/context_provider.py: 97%

**Brakujące pokrycie w context_provider.py**:
- Linia 53: Walidacja parametrów
- Linia 78: Obsługa błędów

**Lista testów**:
1. test_initialization ✅
2. test_get_market_context ✅
3. test_format_market_data ✅
4. test_error_handling ✅
5. test_context_limits ✅
6. test_generate_context_summary ✅
7. test_input_validation ✅
8. test_format_market_data_with_many_indicators ✅
9. test_timeframe_formats ✅
10. test_empty_indicators ✅
11. test_timestamp_validation ✅
12. test_memory_usage ✅
13. test_get_market_context_performance ✅
14. test_format_market_data_performance ✅
15. test_generate_summary_performance ✅
16. test_memory_usage ✅

**Wyniki wydajności**:
- test_generate_summary_performance:
  - Średni czas: 6.40 μs
  - OPS: 156,334 op/s
- test_format_market_data_performance:
  - Średni czas: 15.90 μs
  - OPS: 62,887 op/s
- test_get_market_context_performance:
  - Średni czas: 143.15 μs
  - OPS: 6,986 op/s

**Moduły z zerowym pokryciem**:
- src/backtest/*
- src/connectors/*
- src/trading/*
- src/utils/prompts.py
- src/utils/technical_indicators.py
- src/strategies/* 


# test_basic_strategy.py ❌

**Status**: 8 testów zaliczonych, 6 testów niezaliczonych
**Liczba testów**: 14
**Czas wykonania**: 9.75s

**Pokrycie kodu**:
- Całkowita liczba linii: 2560
- Niepokryte linie: 2016
- Pokrycie całkowite: 21.25%
- Pokrycie src/strategies/basic_strategy.py: 71%

**Brakujące pokrycie w basic_strategy.py**:
- Linie 35-53: Inicjalizacja i walidacja
- Linie 83, 91: Obsługa błędów
- Linie 102-106: Walidacja danych wejściowych
- Linie 133, 137-138: Generowanie sygnałów
- Linie 164-168: Analiza rynku
- Linie 201-203, 227-229: Obsługa przypadków brzegowych
- Linie 253-255, 331-333: Obliczanie pozycji
- Linie 347, 400-402: Wykonywanie sygnałów
- Linie 418-419, 425-426, 432-433: Metryki wydajności
- Linie 439-440, 444-445, 448-449: Obsługa błędów
- Linie 452-453, 456-457, 461-463: Funkcje pomocnicze

**Lista testów**:
1. test_strategy_initialization ✅
2. test_generate_signals_no_position ❌ (AttributeError: 'BasicStrategy' object has no attribute 'logger')
3. test_generate_signals_with_position ❌ (AttributeError: 'BasicStrategy' object has no attribute 'logger')
4. test_generate_signals_error_handling ❌ (AttributeError: 'BasicStrategy' object has no attribute 'logger')
5. test_analyze_market ❌ (AttributeError: 'BasicStrategy' object has no attribute 'logger')
6. test_execute_signals ✅
7. test_calculate_position_size ✅
8. test_calculate_stop_loss ✅
9. test_calculate_take_profit ✅
10. test_strategy_performance_metrics ✅
11. test_strategy_analysis_speed ✅
12. test_strategy_memory_usage ✅
13. test_strategy_recovery ❌ (AssertionError: "'BasicStrategy' object has no attribute 'logger'" != "Connection error")
14. test_strategy_integration ❌ (AttributeError: 'BasicStrategy' object has no attribute 'logger')

**Wyniki wydajności**:
- test_strategy_analysis_speed:
  - Średni czas: 12.96 ms
  - OPS: 77.14 op/s

**Ostrzeżenia**:
1. Nieznany marker pytest.mark.memory
2. Przestarzałe użycie event_loop fixture

**Moduły z zerowym pokryciem**:
- src/backtest/*
- src/connectors/*
- src/rag/*
- src/trading/mt5_handler.py
- src/trading/position_manager.py
- src/trading/position_status.py
- src/trading/trade_type.py
- src/utils/prompts.py 


# test_embeddings_handler.py ✅

**Status**: Wszystkie testy zaliczone
**Liczba testów**: 16
**Czas wykonania**: 106.82s

**Pokrycie kodu**:
- Całkowita liczba linii: 2560
- Niepokryte linie: 2181
- Pokrycie całkowite: 14.80%
- Pokrycie src/rag/embeddings_handler.py: 100%

**Lista testów**:
1. test_initialization ✅
2. test_generate_embeddings ✅
3. test_calculate_similarity ✅
4. test_get_most_similar ✅
5. test_error_handling ✅
6. test_input_validation ✅
7. test_save_embeddings ✅
8. test_load_embeddings ✅
9. test_save_embeddings_errors ✅
10. test_load_embeddings_errors ✅
11. test_generate_embeddings_performance ✅
12. test_memory_usage ✅
13. test_batch_processing_performance ✅
14. test_similarity_calculation_performance ✅
15. test_concurrent_requests_performance ✅
16. test_model_loading_performance ✅

**Wyniki wydajności**:
- test_generate_embeddings_performance:
  - Średni czas: 15.42 ms
  - OPS: 64.85 op/s
- test_batch_processing_performance:
  - Średni czas: 82.31 ms
  - OPS: 12.15 op/s
- test_similarity_calculation_performance:
  - Średni czas: 0.95 ms
  - OPS: 1052.63 op/s

**Moduły z zerowym pokryciem**:
- src/backtest/*
- src/connectors/*
- src/trading/*
- src/rag/market_memory.py
- src/rag/context_provider.py
- src/utils/prompts.py
- src/utils/technical_indicators.py
- src/strategies/* 


# test_mt5_handler.py ⚠️

**Status**: 24 testy zaliczone, 1 test pominięty
**Liczba testów**: 25
**Czas wykonania**: 3.67s

**Pokrycie kodu**:
- Całkowita liczba linii: 2560
- Niepokryte linie: 2081
- Pokrycie całkowite: 18.71%
- Pokrycie src/trading/mt5_handler.py: 93%

**Brakujące pokrycie w mt5_handler.py**:
- Linia 85: Walidacja parametrów
- Linie 128, 130: Obsługa błędów
- Linie 269-277: Obsługa przypadków brzegowych
- Linia 375: Funkcje pomocnicze

**Lista testów**:
1. test_initialization ✅
2. test_initialization_error ✅
3. test_open_position ✅
4. test_open_position_error ✅
5. test_close_position ✅
6. test_close_position_no_position ✅
7. test_get_current_price ✅
8. test_get_historical_data ✅
9. test_get_historical_data_error ✅
10. test_get_account_info ✅
11. test_get_positions ✅
12. test_convert_timeframe ✅
13. test_strategy_execution_speed ✅
14. test_memory_usage ✅
15. test_full_trading_cycle ⚠️ (pominięty - brak obsługi async)
16. test_error_handling_under_load ✅

**Wyniki wydajności**:
- test_strategy_execution_speed:
  - Średni czas: 2.68 μs
  - OPS: 372,756 op/s

**Ostrzeżenia**:
1. Nieznany marker pytest.mark.memory
2. Przestarzałe użycie event_loop fixture
3. Coroutine 'MT5Handler.get_current_price' nigdy nie został awaited
4. Brak natywnego wsparcia dla async def functions

**Moduły z zerowym pokryciem**:
- src/backtest/*
- src/connectors/*
- src/rag/*
- src/utils/prompts.py
- src/utils/technical_indicators.py
- src/strategies/* 

# test_market_memory.py ✅

**Status**: Wszystkie testy zaliczone
**Liczba testów**: 49
**Czas wykonania**: 70.68s

**Pokrycie kodu**:
- Całkowita liczba linii: 2560
- Niepokryte linie: 2061
- Pokrycie całkowite: 19.49%
- Pokrycie src/rag/market_memory.py: 94%

**Brakujące pokrycie w market_memory.py**:
- Linia 82: Walidacja parametrów
- Linia 275: Obsługa błędów
- Linia 286: Walidacja danych wejściowych
- Linie 316-318: Obsługa przypadków brzegowych
- Linie 341-342: Obsługa błędów zamykania
- Linie 355-357: Funkcje pomocnicze

**Lista testów**:
1. test_inicjalizacja_market_memory ✅
2. test_dodawanie_danych_rynkowych ✅
3. test_walidacja_danych_wejsciowych ✅
4. test_walidacja_wyszukiwania ✅
5. test_walidacja_usuwania ✅
6. test_wyszukiwanie_danych ✅
7. test_usuwanie_danych ✅
8. test_duze_dane ✅
9. test_wiele_symboli_i_timeframes ✅
10. test_operacje_wspolbiezne ✅
11. test_zuzycie_pamieci ✅
12. test_obsluga_bledow_inicjalizacji ✅
13. test_obsluga_bledow_dodawania ✅
14. test_obsluga_bledow_wyszukiwania ✅
15. test_obsluga_bledow_usuwania ✅
16. test_zamykanie_polaczen ✅
17. test_bledy_dostepu_do_katalogu ✅
18. test_bledy_dodawania_do_kolekcji ✅
19. test_bledy_przetwarzania_wynikow ✅
20. test_bledy_usuwania_danych ✅
21. test_bledy_zamykania_polaczen ✅
22. test_bledy_wyszukiwania ✅
23. test_bledy_przetwarzania_daty ✅
24. test_bledy_usuwania_szczegolowe ✅
25. test_bledy_zamykania_polaczen_szczegolowe ✅
26. test_bledy_zamykania_polaczen_pelne ✅
27. test_sciezka_walidacji_katalogu ✅
28. test_obsluga_bledow_dodawania_szczegolowa ✅
29. test_obsluga_bledow_wyszukiwania_szczegolowa ✅
30. test_obsluga_bledow_usuwania_szczegolowa ✅
31. test_obsluga_bledow_zamykania_szczegolowa ✅
32. test_obsluga_bledow_dodawania_pelna ✅
33. test_obsluga_bledow_wyszukiwania_pelna ✅
34. test_obsluga_bledow_usuwania_pelna ✅
35. test_obsluga_bledow_destruktora ✅
36. test_obsluga_bledow_usuwania_wsadow ✅
37. test_obsluga_bledow_przetwarzania_daty_dokumentu ✅
38. test_obsluga_bledow_usuwania_kolekcji_szczegolowa ✅
39. test_obsluga_bledow_destruktora_szczegolowa ✅
40. test_obsluga_bledow_zamykania_polaczen_z_wyjatkiem ✅
41. test_obsluga_bledow_usuwania_kolekcji_w_close ✅
42. test_obsluga_bledow_parsowania_daty_indeksu_szczegolowa ✅
43. test_obsluga_bledow_wyszukiwania_z_nieprawidlowymi_metadanymi_szczegolowa ✅
44. test_obsluga_bledow_zamykania_polaczen_szczegolowa_z_wyjatkiem ✅
45. test_obsluga_bledow_destruktora_szczegolowa_z_wyjatkiem ✅
46. test_obsluga_bledow_usuwania_kolekcji_w_close_szczegolowa ✅
47. test_obsluga_bledow_wyszukiwania_z_pustymi_wynikami_szczegolowa ✅
48. test_obsluga_bledow_pelna_sciezka ✅
49. test_obsluga_bledow_pelna_sciezka_z_nieprawidlowym_indeksem ✅

**Moduły z zerowym pokryciem**:
- src/backtest/*
- src/connectors/*
- src/trading/*
- src/rag/context_provider.py
- src/rag/embeddings_handler.py
- src/utils/prompts.py
- src/utils/technical_indicators.py
- src/strategies/* 


# Wyniki testów pojedynczych plików 🧪

## test_technical_indicators.py ✅

**Status**: Wszystkie testy zaliczone
**Liczba testów**: 31
**Czas wykonania**: 1.96s

**Pokrycie kodu**:
- Całkowita liczba linii: 2560
- Niepokryte linie: 2007
- Pokrycie całkowite: 21.60%
- Pokrycie src/utils/technical_indicators.py: 84%

**Brakujące pokrycie w technical_indicators.py**:
- Linie 50-57: Walidacja parametrów
- Linia 62: Obsługa błędów
- Linia 76, 98, 131: Walidacja danych wejściowych
- Linie 190, 192, 227, 262, 313, 360, 393, 411, 415: Obsługa przypadków brzegowych
- Linie 501-519: Wykrywanie wzorców
- Linie 534-548: Wykrywanie dywergencji
- Linie 566-579: Funkcje pomocnicze

**Lista testów**:
1. test_init_default_params ✅
2. test_init_custom_params ✅
3. test_calculate_all_invalid_input ✅
4. test_calculate_all_missing_columns ✅
5. test_calculate_all_empty_dataframe ✅
6. test_moving_averages ✅
7. test_rsi ✅
8. test_bollinger_bands ✅
9. test_macd ✅
10. test_stochastic ✅
11. test_atr ✅
12. test_momentum ✅
13. test_support_resistance ✅
14. test_detect_patterns ✅
15. test_detect_divergence ✅
16. test_calculate_sma ✅
17. test_calculate_ema ✅
18. test_calculate_rsi ✅
19. test_calculate_macd ✅
20. test_calculate_bollinger_bands ✅
21. test_calculate_stochastic ✅
22. test_calculate_atr ✅
23. test_calculate_adx ✅
24. test_calculate_pivot_points ✅
25. test_detect_patterns_detailed ✅
26. test_detect_patterns_validation ✅
27. test_detect_patterns_edge_cases ✅
28. test_input_validation_edge_cases ✅
29. test_pivot_points_detailed ✅
30. test_divergence_detection ✅
31. test_helper_functions_edge_cases ✅

**Moduły z zerowym pokryciem**:
- src/backtest/*
- src/connectors/*
- src/trading/*
- src/rag/*
- src/utils/prompts.py
- src/interfaces/connectors.py
- src/strategies/* 


# test_mt5_connector.py ✅

**Status**: Wszystkie 10 testów zaliczonych
**Liczba testów**: 10
**Czas wykonania**: 1.76s

**Pokrycie kodu**:
- Całkowita liczba linii: 2560
- Niepokryte linie: 2184
- Pokrycie całkowite: 14.69%
- Pokrycie src/connectors/mt5_connector.py: 100%

**Lista testów**:
1. test_connect_success ✅
2. test_connect_initialize_failure ✅
3. test_connect_login_failure ✅
4. test_disconnect ✅
5. test_get_account_info_success ✅
6. test_get_account_info_not_connected ✅
7. test_get_account_info_failure ✅
8. test_get_symbols_success ✅
9. test_get_symbols_not_connected ✅
10. test_get_symbols_failure ✅

**Ostrzeżenia**:
1. Przestarzałe użycie event_loop fixture

**Moduły z zerowym pokryciem**:
- src/backtest/*
- src/rag/*
- src/trading/mt5_handler.py
- src/trading/position_manager.py
- src/trading/position_status.py
- src/trading/trade_type.py
- src/utils/prompts.py
- src/utils/technical_indicators.py

# test_anthropic_connector.py ✅

**Status**: Wszystkie 6 testów zaliczonych
**Liczba testów**: 6
**Czas wykonania**: 2.68s

**Pokrycie kodu**:
- Całkowita liczba linii: 2560
- Niepokryte linie: 2226
- Pokrycie całkowite: 13.05%
- Pokrycie src/connectors/anthropic_connector.py: 100%

**Lista testów**:
1. test_initialization ✅
2. test_initialization_missing_api_key ✅
3. test_analyze_market_conditions_success ✅
4. test_analyze_market_conditions_auth_error ✅
5. test_analyze_market_conditions_general_error ✅
6. test_analyze_market_conditions_invalid_template ✅

**Ostrzeżenia**:
1. Przestarzałe użycie event_loop fixture

**Moduły z zerowym pokryciem**:
- src/backtest/*
- src/connectors/mt5_connector.py
- src/connectors/ollama_connector.py
- src/trading/*
- src/rag/*
- src/utils/prompts.py
- src/utils/technical_indicators.py
- src/strategies/* 


