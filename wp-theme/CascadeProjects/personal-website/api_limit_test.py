import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import json
import random

import aiohttp
from prometheus_client import Counter, Gauge, Histogram
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIError(Exception):
    """API-virhe"""
    def __init__(
        self,
        message: str,
        status_code: int,
        retry_after: Optional[int] = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after

class RateLimitError(APIError):
    """Rate limit -virhe"""
    pass

class TokenLimitError(APIError):
    """Token limit -virhe"""
    pass

class APILimitTest:
    """API-rajojen testaaja"""
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        rate_limit: int = 100,    # kutsu/min
        token_limit: int = 1000,  # tokenia/min
        max_retries: int = 3,
        backoff_factor: float = 1.5
    ):
        """
        Alusta testaaja
        
        Args:
            base_url: API:n perus-URL
            api_key: API-avain
            rate_limit: Kutsujen raja
            token_limit: Tokenien raja
            max_retries: Uudelleenyritykset
            backoff_factor: Backoff-kerroin
        """
        self.base_url = base_url
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.token_limit = token_limit
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        # Metriikat
        self.request_counter = Counter(
            'api_requests_total',
            'Total API requests',
            ['endpoint', 'status']
        )
        self.token_counter = Counter(
            'api_tokens_total',
            'Total API tokens',
            ['endpoint']
        )
        self.latency_histogram = Histogram(
            'api_latency_seconds',
            'API latency',
            ['endpoint']
        )
        self.retry_counter = Counter(
            'api_retries_total',
            'Total API retries',
            ['endpoint']
        )
        self.error_counter = Counter(
            'api_errors_total',
            'Total API errors',
            ['endpoint', 'type']
        )
    
    async def test_rate_limits(
        self,
        endpoint: str,
        requests_per_second: int,
        duration: int
    ) -> Dict:
        """
        Testaa rate limitejä
        
        Args:
            endpoint: API-endpoint
            requests_per_second: Kutsujen määrä/s
            duration: Testin kesto sekunteina
        
        Returns:
            Dict: Testitulokset
        """
        logger.info(
            f"Testing rate limits for {endpoint} "
            f"({requests_per_second} req/s for {duration}s)"
        )
        
        results = {
            "requests": 0,
            "success": 0,
            "rate_limits": 0,
            "other_errors": 0,
            "retries": 0,
            "latencies": []
        }
        
        start_time = datetime.now()
        
        while (
            datetime.now() - start_time
        ).total_seconds() < duration:
            tasks = [
                self._make_request(endpoint, results)
                for _ in range(requests_per_second)
            ]
            
            await asyncio.gather(*tasks)
            await asyncio.sleep(1)
        
        # Laske tilastot
        avg_latency = (
            sum(results["latencies"]) /
            len(results["latencies"])
            if results["latencies"]
            else 0
        )
        
        return {
            **results,
            "duration": duration,
            "avg_latency": avg_latency,
            "success_rate": (
                results["success"] /
                results["requests"] * 100
                if results["requests"]
                else 0
            )
        }
    
    async def test_token_limits(
        self,
        endpoint: str,
        tokens_per_request: int,
        requests_per_second: int,
        duration: int
    ) -> Dict:
        """
        Testaa token limitejä
        
        Args:
            endpoint: API-endpoint
            tokens_per_request: Tokenien määrä/kutsu
            requests_per_second: Kutsujen määrä/s
            duration: Testin kesto sekunteina
        
        Returns:
            Dict: Testitulokset
        """
        logger.info(
            f"Testing token limits for {endpoint} "
            f"({tokens_per_request} tokens/req, "
            f"{requests_per_second} req/s for {duration}s)"
        )
        
        results = {
            "requests": 0,
            "success": 0,
            "token_limits": 0,
            "other_errors": 0,
            "retries": 0,
            "tokens_used": 0,
            "latencies": []
        }
        
        start_time = datetime.now()
        
        while (
            datetime.now() - start_time
        ).total_seconds() < duration:
            tasks = [
                self._make_request(
                    endpoint,
                    results,
                    tokens_per_request
                )
                for _ in range(requests_per_second)
            ]
            
            await asyncio.gather(*tasks)
            await asyncio.sleep(1)
        
        # Laske tilastot
        avg_latency = (
            sum(results["latencies"]) /
            len(results["latencies"])
            if results["latencies"]
            else 0
        )
        
        return {
            **results,
            "duration": duration,
            "avg_latency": avg_latency,
            "success_rate": (
                results["success"] /
                results["requests"] * 100
                if results["requests"]
                else 0
            ),
            "tokens_per_second": (
                results["tokens_used"] / duration
            )
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _make_request(
        self,
        endpoint: str,
        results: Dict,
        tokens: int = 0
    ):
        """
        Tee API-kutsu
        
        Args:
            endpoint: API-endpoint
            results: Tulokset
            tokens: Tokenien määrä
        """
        url = f"{self.base_url}{endpoint}"
        results["requests"] += 1
        
        try:
            start_time = datetime.now()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "tokens": tokens,
                        "data": "x" * random.randint(10, 100)
                    }
                ) as response:
                    duration = (
                        datetime.now() - start_time
                    ).total_seconds()
                    
                    results["latencies"].append(duration)
                    
                    # Päivitä metriikat
                    self.latency_histogram.labels(
                        endpoint=endpoint
                    ).observe(duration)
                    
                    if response.status == 429:  # Rate limit
                        results["rate_limits"] += 1
                        self.error_counter.labels(
                            endpoint=endpoint,
                            type="rate_limit"
                        ).inc()
                        
                        retry_after = int(
                            response.headers.get(
                                "Retry-After",
                                "5"
                            )
                        )
                        raise RateLimitError(
                            "Rate limit exceeded",
                            429,
                            retry_after
                        )
                    
                    elif response.status == 429:  # Token limit
                        results["token_limits"] += 1
                        self.error_counter.labels(
                            endpoint=endpoint,
                            type="token_limit"
                        ).inc()
                        
                        retry_after = int(
                            response.headers.get(
                                "Retry-After",
                                "60"
                            )
                        )
                        raise TokenLimitError(
                            "Token limit exceeded",
                            429,
                            retry_after
                        )
                    
                    elif response.status != 200:
                        results["other_errors"] += 1
                        self.error_counter.labels(
                            endpoint=endpoint,
                            type="other"
                        ).inc()
                        
                        raise APIError(
                            f"API error: {response.status}",
                            response.status
                        )
                    
                    results["success"] += 1
                    if tokens > 0:
                        results["tokens_used"] += tokens
                    
                    self.request_counter.labels(
                        endpoint=endpoint,
                        status="success"
                    ).inc()
                    
                    if tokens > 0:
                        self.token_counter.labels(
                            endpoint=endpoint
                        ).inc(tokens)
        
        except (RateLimitError, TokenLimitError) as e:
            # Odota retry-after
            if e.retry_after:
                await asyncio.sleep(e.retry_after)
            
            results["retries"] += 1
            self.retry_counter.labels(
                endpoint=endpoint
            ).inc()
            
            raise  # Uudelleenyritys
        
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise
    
    def generate_report(
        self,
        rate_results: Dict,
        token_results: Dict
    ) -> str:
        """
        Generoi raportti
        
        Args:
            rate_results: Rate limit -tulokset
            token_results: Token limit -tulokset
        
        Returns:
            str: Markdown-muotoinen raportti
        """
        report = """# API-rajojen testiraportti

## Rate Limit -testit

"""
        
        report += f"- Kesto: {rate_results['duration']}s\n"
        report += f"- Kutsuja: {rate_results['requests']}\n"
        report += f"- Onnistuneet: {rate_results['success']}\n"
        report += f"- Rate limitit: {rate_results['rate_limits']}\n"
        report += f"- Muut virheet: {rate_results['other_errors']}\n"
        report += f"- Uudelleenyritykset: {rate_results['retries']}\n"
        report += f"- Onnistumisprosentti: {rate_results['success_rate']:.1f}%\n"
        report += f"- Keskimääräinen latenssi: {rate_results['avg_latency']:.3f}s\n\n"
        
        report += "## Token Limit -testit\n\n"
        
        report += f"- Kesto: {token_results['duration']}s\n"
        report += f"- Kutsuja: {token_results['requests']}\n"
        report += f"- Onnistuneet: {token_results['success']}\n"
        report += f"- Token limitit: {token_results['token_limits']}\n"
        report += f"- Muut virheet: {token_results['other_errors']}\n"
        report += f"- Uudelleenyritykset: {token_results['retries']}\n"
        report += f"- Käytetyt tokenit: {token_results['tokens_used']}\n"
        report += f"- Tokenia/sekunti: {token_results['tokens_per_second']:.1f}\n"
        report += f"- Onnistumisprosentti: {token_results['success_rate']:.1f}%\n"
        report += f"- Keskimääräinen latenssi: {token_results['avg_latency']:.3f}s\n"
        
        return report

async def main():
    """Testaa API-rajoja"""
    # Alusta testaaja
    tester = APILimitTest(
        base_url="https://api.example.com",
        api_key="your-api-key",
        rate_limit=100,
        token_limit=1000
    )
    
    # Testaa rate limitejä
    rate_results = await tester.test_rate_limits(
        endpoint="/test",
        requests_per_second=20,
        duration=30
    )
    
    # Testaa token limitejä
    token_results = await tester.test_token_limits(
        endpoint="/test",
        tokens_per_request=50,
        requests_per_second=10,
        duration=30
    )
    
    # Tulosta raportti
    print("\nTest Report:")
    print(tester.generate_report(
        rate_results,
        token_results
    ))

if __name__ == "__main__":
    asyncio.run(main())
