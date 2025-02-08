"""
Windsurf monitorointi.
Tämä moduuli vastaa API-kutsujen ja kustannusten monitoroinnista.
"""

import os
import json
import logging
import asyncio
import sqlite3
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import anthropic
from anthropic import Client
from logging.handlers import RotatingFileHandler
from path_utils import PathManager

# Konfiguroi lokitus
def setup_logging(
    log_dir: str = "logs",
    max_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """Konfiguroi lokitus"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Pääloki
    main_handler = RotatingFileHandler(
        os.path.join(log_dir, "windsurf.log"),
        maxBytes=max_size,
        backupCount=backup_count
    )
    main_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # API-loki
    api_handler = RotatingFileHandler(
        os.path.join(log_dir, "api.log"),
        maxBytes=max_size,
        backupCount=backup_count
    )
    api_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(message)s'
    ))
    
    # Virheloki
    error_handler = RotatingFileHandler(
        os.path.join(log_dir, "errors.log"),
        maxBytes=max_size,
        backupCount=backup_count
    )
    error_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
        'Exception: %(exc_info)s'
    ))
    error_handler.setLevel(logging.ERROR)
    
    # Kustannusloki
    cost_handler = RotatingFileHandler(
        os.path.join(log_dir, "costs.log"),
        maxBytes=max_size,
        backupCount=backup_count
    )
    cost_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(message)s'
    ))
    
    # Konfiguroi loggerit
    logging.basicConfig(
        level=logging.INFO,
        handlers=[main_handler]
    )
    
    api_logger = logging.getLogger('api')
    api_logger.addHandler(api_handler)
    api_logger.propagate = False
    
    error_logger = logging.getLogger('error')
    error_logger.addHandler(error_handler)
    error_logger.propagate = False
    
    cost_logger = logging.getLogger('cost')
    cost_logger.addHandler(cost_handler)
    cost_logger.propagate = False

class APIMetricsDB:
    """API-metriikoiden tietokanta"""
    
    def __init__(self, db_path: str = "metrics.db"):
        """
        Alusta tietokanta
        
        Args:
            db_path: Tietokannan polku
        """
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Alusta tietokanta"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # API-kutsut
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_calls (
                    id INTEGER PRIMARY KEY,
                    timestamp DATETIME,
                    model TEXT,
                    endpoint TEXT,
                    tokens_in INTEGER,
                    tokens_out INTEGER,
                    duration REAL,
                    cost REAL,
                    success BOOLEAN,
                    error_type TEXT
                )
            """)
            
            # Kustannukset per malli
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_costs (
                    id INTEGER PRIMARY KEY,
                    date DATE,
                    model TEXT,
                    total_tokens INTEGER,
                    total_cost REAL
                )
            """)
            
            # Rate limit tilastot
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rate_limits (
                    id INTEGER PRIMARY KEY,
                    timestamp DATETIME,
                    endpoint TEXT,
                    limit_type TEXT,
                    remaining INTEGER
                )
            """)
            
            conn.commit()
    
    def log_api_call(
        self,
        model: str,
        endpoint: str,
        tokens_in: int,
        tokens_out: int,
        duration: float,
        cost: float,
        success: bool,
        error_type: Optional[str] = None
    ):
        """Kirjaa API-kutsu"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO api_calls (
                    timestamp, model, endpoint, tokens_in,
                    tokens_out, duration, cost, success, error_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(), model, endpoint, tokens_in,
                tokens_out, duration, cost, success, error_type
            ))
            conn.commit()
    
    def update_model_costs(
        self,
        model: str,
        tokens: int,
        cost: float
    ):
        """Päivitä mallin kustannukset"""
        today = datetime.now().date()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Hae olemassaoleva rivi
            cursor.execute("""
                SELECT total_tokens, total_cost
                FROM model_costs
                WHERE date = ? AND model = ?
            """, (today, model))
            
            row = cursor.fetchone()
            
            if row:
                # Päivitä olemassaoleva
                cursor.execute("""
                    UPDATE model_costs
                    SET total_tokens = total_tokens + ?,
                        total_cost = total_cost + ?
                    WHERE date = ? AND model = ?
                """, (tokens, cost, today, model))
            else:
                # Lisää uusi
                cursor.execute("""
                    INSERT INTO model_costs (
                        date, model, total_tokens, total_cost
                    ) VALUES (?, ?, ?, ?)
                """, (today, model, tokens, cost))
            
            conn.commit()
    
    def log_rate_limit(
        self,
        endpoint: str,
        limit_type: str,
        remaining: int
    ):
        """Kirjaa rate limit"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO rate_limits (
                    timestamp, endpoint, limit_type, remaining
                ) VALUES (?, ?, ?, ?)
            """, (datetime.now(), endpoint, limit_type, remaining))
            conn.commit()
    
    def get_daily_costs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Hae päivittäiset kustannukset"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT date, model, total_tokens, total_cost
                FROM model_costs
                WHERE date BETWEEN ? AND ?
                ORDER BY date DESC, model
            """, (start_date.date(), end_date.date()))
            
            return [{
                "date": row[0],
                "model": row[1],
                "tokens": row[2],
                "cost": row[3]
            } for row in cursor.fetchall()]
    
    def get_api_stats(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Hae API-tilastot"""
        since = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Kokonaismäärät
            cursor.execute("""
                SELECT COUNT(*),
                       SUM(CASE WHEN success THEN 1 ELSE 0 END),
                       AVG(duration),
                       SUM(tokens_in + tokens_out),
                       SUM(cost)
                FROM api_calls
                WHERE timestamp > ?
            """, (since,))
            
            totals = cursor.fetchone()
            
            # Virheet per tyyppi
            cursor.execute("""
                SELECT error_type, COUNT(*)
                FROM api_calls
                WHERE timestamp > ? AND NOT success
                GROUP BY error_type
            """, (since,))
            
            errors = {
                row[0]: row[1]
                for row in cursor.fetchall()
            }
            
            return {
                "total_calls": totals[0],
                "successful_calls": totals[1],
                "avg_duration": totals[2],
                "total_tokens": totals[3],
                "total_cost": totals[4],
                "errors": errors
            }

class WindsurfMonitor:
    """Windsurf monitorointi"""
    
    def __init__(
        self,
        db_path: str = "metrics.db",
        log_dir: str = "logs",
        config_file: str = "config.json"
    ):
        """
        Alusta monitorointi
        
        Args:
            db_path: Tietokannan polku
            log_dir: Lokihakemisto
            config_file: Konfiguraatiotiedoston polku
        """
        self.path_manager = PathManager(config_file)
        self.config = self._load_config()
        setup_logging(log_dir)
        self.db = APIMetricsDB(db_path)
        self.logger = logging.getLogger(__name__)
        self.api_logger = logging.getLogger('api')
        self.error_logger = logging.getLogger('error')
        self.cost_logger = logging.getLogger('cost')
    
    def _load_config(self) -> Dict[str, Any]:
        """Lataa konfiguraatio"""
        return json.loads(self.path_manager.config_file.read_text())
        
    def calculate_cost(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int
    ) -> float:
        """
        Laske API-kutsun kustannus
        
        Args:
            model: Mallin nimi
            tokens_in: Input tokenit
            tokens_out: Output tokenit
        
        Returns:
            float: Kustannus dollareina
        """
        # Claude-3 Opus hinnat
        prices = {
            "claude-3-opus-20240229": {
                "input": 0.015,   # per 1K tokenia
                "output": 0.075   # per 1K tokenia
            }
        }
        
        if model not in prices:
            return 0.0
        
        price = prices[model]
        return (
            (tokens_in / 1000) * price["input"] +
            (tokens_out / 1000) * price["output"]
        )
    
    async def monitor_api_call(
        self,
        func: callable,
        model: str,
        endpoint: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Monitoroi API-kutsua
        
        Args:
            func: API-funktio
            model: Mallin nimi
            endpoint: API-endpoint
            *args: Funktion argumentit
            **kwargs: Funktion avainsana-argumentit
        
        Returns:
            Any: API-kutsun tulos
        """
        start_time = datetime.now()
        success = False
        error_type = None
        tokens_in = 0
        tokens_out = 0
        
        try:
            # Suorita API-kutsu
            result = await func(*args, **kwargs)
            
            # Kerää token määrät
            if hasattr(result, "usage"):
                tokens_in = result.usage.input_tokens
                tokens_out = result.usage.output_tokens
            
            success = True
            
            # Loki
            self.api_logger.info(
                f"API-kutsu onnistui: {model} {endpoint} "
                f"({tokens_in + tokens_out} tokenia)"
            )
            
            return result
        
        except Exception as e:
            error_type = type(e).__name__
            self.error_logger.error(
                f"API-virhe: {str(e)}",
                exc_info=True
            )
            raise
        
        finally:
            # Laske kesto ja kustannus
            duration = (datetime.now() - start_time).total_seconds()
            cost = self.calculate_cost(model, tokens_in, tokens_out)
            
            # Kirjaa metriikka
            self.db.log_api_call(
                model=model,
                endpoint=endpoint,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                duration=duration,
                cost=cost,
                success=success,
                error_type=error_type
            )
            
            if success:
                self.db.update_model_costs(
                    model=model,
                    tokens=tokens_in + tokens_out,
                    cost=cost
                )
                
                self.cost_logger.info(
                    f"Kustannus: ${cost:.4f} "
                    f"({tokens_in + tokens_out} tokenia)"
                )
    
    def print_daily_report(self):
        """Tulosta päivittäinen raportti"""
        stats = self.db.get_api_stats(hours=24)
        costs = self.db.get_daily_costs(
            start_date=datetime.now() - timedelta(days=1)
        )
        
        self.logger.info("\n=== Päivittäinen API-raportti ===")
        self.logger.info(f"Kutsuja yhteensä: {stats['total_calls']}")
        self.logger.info(
            f"Onnistumisprosentti: "
            f"{(stats['successful_calls'] / stats['total_calls'] * 100):.1f}%"
        )
        self.logger.info(
            f"Keskimääräinen kesto: {stats['avg_duration']:.2f}s"
        )
        self.logger.info(f"Tokeneja yhteensä: {stats['total_tokens']}")
        self.logger.info(f"Kustannukset yhteensä: ${stats['total_cost']:.2f}")
        
        if stats['errors']:
            self.logger.info("\nVirheet:")
            for error_type, count in stats['errors'].items():
                self.logger.info(f"- {error_type}: {count}")
        
        self.logger.info("\nKustannukset per malli:")
        for cost in costs:
            self.logger.info(
                f"- {cost['model']}: ${cost['cost']:.2f} "
                f"({cost['tokens']} tokenia)"
            )

async def main():
    """Testaa monitorointia"""
    monitor = WindsurfMonitor()
    
    # Testifunktio
    async def test_api(success: bool = True):
        if not success:
            raise Exception("Testi virhe")
        return type("Result", (), {
            "usage": type("Usage", (), {
                "input_tokens": 100,
                "output_tokens": 50
            })
        })
    
    # Testaa onnistunutta kutsua
    try:
        result = await monitor.monitor_api_call(
            test_api,
            model="claude-3-opus-20240229",
            endpoint="/v1/messages",
            success=True
        )
    except Exception as e:
        print(f"Virhe: {str(e)}")
    
    # Testaa epäonnistunutta kutsua
    try:
        result = await monitor.monitor_api_call(
            test_api,
            model="claude-3-opus-20240229",
            endpoint="/v1/messages",
            success=False
        )
    except Exception as e:
        print(f"Virhe: {str(e)}")
    
    # Tulosta raportti
    monitor.print_daily_report()

if __name__ == "__main__":
    asyncio.run(main())
