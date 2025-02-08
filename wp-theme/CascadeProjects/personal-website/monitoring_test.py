{{ ... }}

class APILimitType(Enum):
    """API-rajoitustyypit"""
    TOKEN = "token"           # Token-rajat
    RATE = "rate"            # Kutsurajat
    CONCURRENT = "concurrent" # Rinnakkaiset kutsut
    QUOTA = "quota"          # Kiintiöt
    BANDWIDTH = "bandwidth"   # Kaistanleveys

@dataclass
class APILimit:
    """API-raja"""
    type: APILimitType
    limit: int
    window: int  # sekunteina
    cooldown: int = 60  # sekunteina

class APILimitTester:
    """API-rajojen testaaja"""
    
    def __init__(
        self,
        limits: Dict[str, APILimit]
    ):
        """
        Alusta testaaja
        
        Args:
            limits: API-rajat
        """
        self.limits = limits
        
        # Metriikat
        self.calls_counter = Counter(
            'api_calls_total',
            'API calls',
            ['endpoint', 'status']
        )
        self.limit_gauge = Gauge(
            'api_limit_remaining',
            'API limit remaining',
            ['endpoint', 'type']
        )
        self.latency_histogram = Histogram(
            'api_latency_seconds',
            'API latency',
            ['endpoint']
        )
        
        # Seuranta
        self.call_history: Dict[str, List[float]] = {}
        self.active_calls: Dict[str, int] = {}
        self.quota_used: Dict[str, int] = {}
        self.bandwidth_used: Dict[str, int] = {}
    
    async def test_limits(
        self,
        endpoint: str,
        test_duration: int = 300,
        burst_size: int = 100,
        burst_delay: int = 10
    ) -> Dict[str, Any]:
        """
        Testaa API-rajoja
        
        Args:
            endpoint: Testattava endpoint
            test_duration: Testin kesto sekunteina
            burst_size: Purskeen koko
            burst_delay: Viive purskeiden välillä
        
        Returns:
            Dict[str, Any]: Tulokset
        """
        logger.info(
            f"Testing API limits for {endpoint}"
        )
        
        results = {
            "endpoint": endpoint,
            "duration": test_duration,
            "limits": {},
            "violations": [],
            "recommendations": []
        }
        
        try:
            start_time = datetime.now()
            
            while (
                datetime.now() - start_time
            ).total_seconds() < test_duration:
                # Suorita purske
                burst_results = await self._run_burst(
                    endpoint,
                    burst_size
                )
                
                # Päivitä tulokset
                for limit_type, status in burst_results.items():
                    if limit_type not in results["limits"]:
                        results["limits"][limit_type] = {
                            "total_calls": 0,
                            "violations": 0,
                            "first_violation": None
                        }
                    
                    results["limits"][limit_type]["total_calls"] += (
                        burst_size
                    )
                    
                    if not status["success"]:
                        results["limits"][limit_type]["violations"] += 1
                        
                        if not results["limits"][limit_type]["first_violation"]:
                            results["limits"][limit_type]["first_violation"] = (
                                status["time"]
                            )
                        
                        results["violations"].append({
                            "type": limit_type,
                            "time": status["time"],
                            "details": status["details"]
                        })
                
                # Odota seuraavaan purskeeseen
                await asyncio.sleep(burst_delay)
            
            # Analysoi tulokset
            self._analyze_results(results)
        
        except Exception as e:
            logger.error(
                f"API limit test failed: {str(e)}"
            )
            results["error"] = str(e)
        
        return results
    
    async def _run_burst(
        self,
        endpoint: str,
        size: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Suorita purske
        
        Args:
            endpoint: Endpoint
            size: Purskeen koko
        
        Returns:
            Dict[str, Dict[str, Any]]: Tulokset
        """
        results = {}
        now = datetime.now()
        
        # Päivitä historia
        if endpoint not in self.call_history:
            self.call_history[endpoint] = []
        
        # Poista vanhat kutsut
        self.call_history[endpoint] = [
            t for t in self.call_history[endpoint]
            if (now - datetime.fromtimestamp(t)).total_seconds() < 3600
        ]
        
        # Tarkista rajat
        for limit_type, limit in self.limits.items():
            status = {
                "success": True,
                "time": now.timestamp(),
                "details": {}
            }
            
            if limit.type == APILimitType.TOKEN:
                # Tarkista token-raja
                status = self._check_token_limit(
                    endpoint,
                    size,
                    limit
                )
            
            elif limit.type == APILimitType.RATE:
                # Tarkista kutsuraja
                status = self._check_rate_limit(
                    endpoint,
                    size,
                    limit
                )
            
            elif limit.type == APILimitType.CONCURRENT:
                # Tarkista rinnakkaisuusraja
                status = self._check_concurrent_limit(
                    endpoint,
                    size,
                    limit
                )
            
            elif limit.type == APILimitType.QUOTA:
                # Tarkista kiintiöraja
                status = self._check_quota_limit(
                    endpoint,
                    size,
                    limit
                )
            
            else:  # BANDWIDTH
                # Tarkista kaistanleveysraja
                status = self._check_bandwidth_limit(
                    endpoint,
                    size,
                    limit
                )
            
            results[limit_type] = status
            
            # Päivitä metriikat
            self.calls_counter.labels(
                endpoint=endpoint,
                status="success" if status["success"] else "violation"
            ).inc(size)
            
            self.limit_gauge.labels(
                endpoint=endpoint,
                type=limit.type.value
            ).set(status["details"].get("remaining", 0))
        
        # Lisää kutsut historiaan
        self.call_history[endpoint].extend(
            [now.timestamp()] * size
        )
        
        return results
    
    def _check_token_limit(
        self,
        endpoint: str,
        size: int,
        limit: APILimit
    ) -> Dict[str, Any]:
        """
        Tarkista token-raja
        
        Args:
            endpoint: Endpoint
            size: Purskeen koko
            limit: Raja
        
        Returns:
            Dict[str, Any]: Tulos
        """
        now = datetime.now()
        window_start = now - timedelta(seconds=limit.window)
        
        # Laske käytetyt tokenit
        used_tokens = sum(
            1 for t in self.call_history[endpoint]
            if t >= window_start.timestamp()
        )
        
        # Tarkista raja
        remaining = limit.limit - used_tokens
        success = remaining >= size
        
        return {
            "success": success,
            "time": now.timestamp(),
            "details": {
                "limit": limit.limit,
                "used": used_tokens,
                "remaining": remaining,
                "window": limit.window
            }
        }
    
    def _check_rate_limit(
        self,
        endpoint: str,
        size: int,
        limit: APILimit
    ) -> Dict[str, Any]:
        """
        Tarkista kutsuraja
        
        Args:
            endpoint: Endpoint
            size: Purskeen koko
            limit: Raja
        
        Returns:
            Dict[str, Any]: Tulos
        """
        now = datetime.now()
        window_start = now - timedelta(seconds=limit.window)
        
        # Laske kutsumäärä
        calls = sum(
            1 for t in self.call_history[endpoint]
            if t >= window_start.timestamp()
        )
        
        # Tarkista raja
        rate = calls / limit.window
        success = rate <= limit.limit
        
        return {
            "success": success,
            "time": now.timestamp(),
            "details": {
                "limit": limit.limit,
                "current_rate": rate,
                "window": limit.window
            }
        }
    
    def _check_concurrent_limit(
        self,
        endpoint: str,
        size: int,
        limit: APILimit
    ) -> Dict[str, Any]:
        """
        Tarkista rinnakkaisuusraja
        
        Args:
            endpoint: Endpoint
            size: Purskeen koko
            limit: Raja
        
        Returns:
            Dict[str, Any]: Tulos
        """
        now = datetime.now()
        
        # Päivitä aktiiviset kutsut
        if endpoint not in self.active_calls:
            self.active_calls[endpoint] = 0
        
        # Tarkista raja
        new_total = self.active_calls[endpoint] + size
        success = new_total <= limit.limit
        
        if success:
            self.active_calls[endpoint] = new_total
        
        return {
            "success": success,
            "time": now.timestamp(),
            "details": {
                "limit": limit.limit,
                "active": self.active_calls[endpoint],
                "new_requests": size
            }
        }
    
    def _check_quota_limit(
        self,
        endpoint: str,
        size: int,
        limit: APILimit
    ) -> Dict[str, Any]:
        """
        Tarkista kiintiöraja
        
        Args:
            endpoint: Endpoint
            size: Purskeen koko
            limit: Raja
        
        Returns:
            Dict[str, Any]: Tulos
        """
        now = datetime.now()
        
        # Päivitä käytetty kiintiö
        if endpoint not in self.quota_used:
            self.quota_used[endpoint] = 0
        
        # Tarkista raja
        remaining = limit.limit - self.quota_used[endpoint]
        success = remaining >= size
        
        if success:
            self.quota_used[endpoint] += size
        
        return {
            "success": success,
            "time": now.timestamp(),
            "details": {
                "limit": limit.limit,
                "used": self.quota_used[endpoint],
                "remaining": remaining
            }
        }
    
    def _check_bandwidth_limit(
        self,
        endpoint: str,
        size: int,
        limit: APILimit
    ) -> Dict[str, Any]:
        """
        Tarkista kaistanleveysraja
        
        Args:
            endpoint: Endpoint
            size: Purskeen koko
            limit: Raja
        
        Returns:
            Dict[str, Any]: Tulos
        """
        now = datetime.now()
        window_start = now - timedelta(seconds=limit.window)
        
        # Päivitä käytetty kaista
        if endpoint not in self.bandwidth_used:
            self.bandwidth_used[endpoint] = 0
        
        # Laske käytetty kaista
        bandwidth = self.bandwidth_used[endpoint] / limit.window
        success = bandwidth <= limit.limit
        
        if success:
            self.bandwidth_used[endpoint] += size * 1000  # 1KB per kutsu
        
        return {
            "success": success,
            "time": now.timestamp(),
            "details": {
                "limit": limit.limit,
                "current_bandwidth": bandwidth,
                "window": limit.window
            }
        }
    
    def _analyze_results(
        self,
        results: Dict[str, Any]
    ):
        """
        Analysoi tulokset
        
        Args:
            results: Tulokset
        """
        for limit_type, stats in results["limits"].items():
            if stats["violations"] > 0:
                # Laske rikkomusaste
                violation_rate = (
                    stats["violations"] * 100 /
                    (stats["total_calls"] / self.limits[limit_type].limit)
                )
                
                if violation_rate > 10:  # >10% rikkomuksia
                    results["recommendations"].append(
                        f"High violation rate ({violation_rate:.1f}%) "
                        f"for {limit_type} limit. Consider:"
                    )
                    
                    if limit_type == "token":
                        results["recommendations"].append(
                            "- Implementing token bucket algorithm"
                        )
                        results["recommendations"].append(
                            "- Adding retry with exponential backoff"
                        )
                    
                    elif limit_type == "rate":
                        results["recommendations"].append(
                            "- Adding request queuing"
                        )
                        results["recommendations"].append(
                            "- Implementing rate limiting"
                        )
                    
                    elif limit_type == "concurrent":
                        results["recommendations"].append(
                            "- Adding connection pooling"
                        )
                        results["recommendations"].append(
                            "- Implementing request batching"
                        )
                    
                    elif limit_type == "quota":
                        results["recommendations"].append(
                            "- Implementing quota tracking"
                        )
                        results["recommendations"].append(
                            "- Adding usage monitoring"
                        )
                    
                    else:  # bandwidth
                        results["recommendations"].append(
                            "- Adding request compression"
                        )
                        results["recommendations"].append(
                            "- Implementing response caching"
                        )

class MonitoringTester:
    def __init__(
        self,
        metrics_port: int = 8000,
        alert_webhook: Optional[str] = None
    ):
        """
        Alusta testaaja
        
        Args:
            metrics_port: Prometheus-portti
            alert_webhook: Hälytys-webhook
        """
        # Vanhat alustukset
        {{ ... }}
        
        # API-rajojen testaaja
        self.api_tester = APILimitTester({
            "token": APILimit(
                type=APILimitType.TOKEN,
                limit=1000,
                window=3600
            ),
            "rate": APILimit(
                type=APILimitType.RATE,
                limit=100,
                window=60
            ),
            "concurrent": APILimit(
                type=APILimitType.CONCURRENT,
                limit=10,
                window=1
            ),
            "quota": APILimit(
                type=APILimitType.QUOTA,
                limit=10000,
                window=86400
            ),
            "bandwidth": APILimit(
                type=APILimitType.BANDWIDTH,
                limit=1024*1024,  # 1MB/s
                window=60
            )
        })

{{ ... }}
