"""
Windsurf testiraportointi.
Tämä moduuli vastaa testitulosten raportoinnista ja analysoinnista.
"""

import os
import json
import logging
import sqlite3
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum, auto

# Konfiguroi lokitus
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """Testin tila"""
    SUCCESS = auto()
    FAILED = auto()
    SKIPPED = auto()
    ERROR = auto()

class TestType(Enum):
    """Testin tyyppi"""
    CODE_GENERATION = "code_generation"
    TEXT_ANALYSIS = "text_analysis"
    QUICK_RESPONSE = "quick_response"
    COMPLEX_ANALYSIS = "complex_analysis"
    BATCH_PROCESSING = "batch_processing"
    PERFORMANCE = "performance"
    SECURITY = "security"

class Priority(Enum):
    """Parannusehdotuksen prioriteetti"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class TestResult:
    """Testitulos"""
    name: str
    status: TestStatus
    type: TestType
    details: str
    error: Optional[str] = None
    duration: Optional[float] = None
    timestamp: datetime = datetime.now()

@dataclass
class Improvement:
    """Parannusehdotus"""
    priority: Priority
    suggestion: str
    actionable: bool
    automated: bool
    affected_tests: List[str]
    estimated_effort: str

class TestReportGenerator:
    """Testiraporttien generaattori"""
    
    def __init__(self, db_path: str = "test_results.db"):
        """
        Alusta generaattori
        
        Args:
            db_path: Tietokannan polku
        """
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Alusta tietokanta"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Testitulokset
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    status TEXT,
                    type TEXT,
                    details TEXT,
                    error TEXT,
                    duration REAL,
                    timestamp DATETIME
                )
            """)
            
            # Parannusehdotukset
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS improvements (
                    id INTEGER PRIMARY KEY,
                    priority TEXT,
                    suggestion TEXT,
                    actionable BOOLEAN,
                    automated BOOLEAN,
                    estimated_effort TEXT,
                    timestamp DATETIME
                )
            """)
            
            # Parannusehdotusten ja testien linkitys
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS improvement_tests (
                    improvement_id INTEGER,
                    test_name TEXT,
                    FOREIGN KEY(improvement_id) REFERENCES improvements(id)
                )
            """)
            
            conn.commit()
    
    def save_test_result(self, result: TestResult):
        """Tallenna testitulos"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO test_results (
                    name, status, type, details,
                    error, duration, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                result.name,
                result.status.name,
                result.type.value,
                result.details,
                result.error,
                result.duration,
                result.timestamp
            ))
            conn.commit()
    
    def save_improvement(self, improvement: Improvement):
        """Tallenna parannusehdotus"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tallenna parannusehdotus
            cursor.execute("""
                INSERT INTO improvements (
                    priority, suggestion, actionable,
                    automated, estimated_effort, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                improvement.priority.value,
                improvement.suggestion,
                improvement.actionable,
                improvement.automated,
                improvement.estimated_effort,
                datetime.now()
            ))
            
            improvement_id = cursor.lastrowid
            
            # Tallenna linkitykset testeihin
            for test_name in improvement.affected_tests:
                cursor.execute("""
                    INSERT INTO improvement_tests (
                        improvement_id, test_name
                    ) VALUES (?, ?)
                """, (improvement_id, test_name))
            
            conn.commit()
    
    def analyze_results(
        self,
        results: List[TestResult]
    ) -> List[Improvement]:
        """
        Analysoi testitulokset ja generoi parannusehdotukset
        
        Args:
            results: Testitulokset
        
        Returns:
            List[Improvement]: Parannusehdotukset
        """
        improvements = []
        
        # Ryhmittele testit tyypin mukaan
        tests_by_type = {}
        for result in results:
            if result.type not in tests_by_type:
                tests_by_type[result.type] = []
            tests_by_type[result.type].append(result)
        
        # Analysoi epäonnistuneet testit
        failed_tests = [
            r for r in results
            if r.status == TestStatus.FAILED
        ]
        
        if failed_tests:
            # Ryhmittele virheet
            errors_by_type = {}
            for test in failed_tests:
                if test.error not in errors_by_type:
                    errors_by_type[test.error] = []
                errors_by_type[test.error].append(test)
            
            # Generoi parannusehdotukset virhetyypeittäin
            for error_type, tests in errors_by_type.items():
                if "timeout" in error_type.lower():
                    improvements.append(Improvement(
                        priority=Priority.HIGH,
                        suggestion="Optimoi suorituskykyä timeout-virheiden välttämiseksi",
                        actionable=True,
                        automated=True,
                        affected_tests=[t.name for t in tests],
                        estimated_effort="2-4h"
                    ))
                elif "token" in error_type.lower():
                    improvements.append(Improvement(
                        priority=Priority.HIGH,
                        suggestion="Optimoi token käyttöä ja jaa isot tehtävät osiin",
                        actionable=True,
                        automated=True,
                        affected_tests=[t.name for t in tests],
                        estimated_effort="4-8h"
                    ))
        
        # Analysoi suorituskyky
        performance_tests = tests_by_type.get(TestType.PERFORMANCE, [])
        if performance_tests:
            slow_tests = [
                t for t in performance_tests
                if t.duration and t.duration > 1.0
            ]
            if slow_tests:
                improvements.append(Improvement(
                    priority=Priority.MEDIUM,
                    suggestion="Optimoi hitaita testejä välimuistilla ja eräajolla",
                    actionable=True,
                    automated=True,
                    affected_tests=[t.name for t in slow_tests],
                    estimated_effort="4-6h"
                ))
        
        # Analysoi turvallisuus
        security_tests = tests_by_type.get(TestType.SECURITY, [])
        if security_tests:
            failed_security = [
                t for t in security_tests
                if t.status == TestStatus.FAILED
            ]
            if failed_security:
                improvements.append(Improvement(
                    priority=Priority.HIGH,
                    suggestion="Korjaa tietoturvaongelmat ja päivitä testit",
                    actionable=True,
                    automated=False,
                    affected_tests=[t.name for t in failed_security],
                    estimated_effort="8-16h"
                ))
        
        return improvements
    
    def generate_report(
        self,
        results: List[TestResult]
    ) -> Dict[str, Any]:
        """
        Generoi testiraportti
        
        Args:
            results: Testitulokset
        
        Returns:
            Dict[str, Any]: Raportti
        """
        # Tallenna tulokset
        for result in results:
            self.save_test_result(result)
        
        # Analysoi ja tallenna parannusehdotukset
        improvements = self.analyze_results(results)
        for improvement in improvements:
            self.save_improvement(improvement)
        
        # Kokoa raportti
        successful = [
            r for r in results
            if r.status == TestStatus.SUCCESS
        ]
        failed = [
            r for r in results
            if r.status == TestStatus.FAILED
        ]
        
        return {
            "timestamp": datetime.now(),
            "summary": {
                "total_tests": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(results) * 100
            },
            "successful_tests": [{
                "name": r.name,
                "type": r.type.value,
                "details": r.details,
                "duration": r.duration
            } for r in successful],
            "failed_tests": [{
                "name": r.name,
                "type": r.type.value,
                "details": r.details,
                "error": r.error,
                "duration": r.duration
            } for r in failed],
            "improvements": [{
                "priority": i.priority.value,
                "suggestion": i.suggestion,
                "actionable": i.actionable,
                "automated": i.automated,
                "affected_tests": i.affected_tests,
                "estimated_effort": i.estimated_effort
            } for i in improvements]
        }

def main():
    """Testaa raportointia"""
    # Luo testidataa
    results = [
        TestResult(
            name="Code Generation Test",
            status=TestStatus.SUCCESS,
            type=TestType.CODE_GENERATION,
            details="Koodi generoitiin oikein",
            duration=0.5
        ),
        TestResult(
            name="Text Analysis Test",
            status=TestStatus.FAILED,
            type=TestType.TEXT_ANALYSIS,
            details="Virhe analyysissä",
            error="Timeout",
            duration=2.5
        ),
        TestResult(
            name="Quick Response Test",
            status=TestStatus.SUCCESS,
            type=TestType.QUICK_RESPONSE,
            details="Vastaus saatiin nopeasti",
            duration=0.1
        ),
        TestResult(
            name="Complex Analysis Test",
            status=TestStatus.FAILED,
            type=TestType.COMPLEX_ANALYSIS,
            details="Ei riittävästi kontekstia",
            error="Token limit exceeded",
            duration=1.5
        ),
        TestResult(
            name="Security Test",
            status=TestStatus.FAILED,
            type=TestType.SECURITY,
            details="API-avain vuoto",
            error="Security vulnerability",
            duration=0.3
        )
    ]
    
    # Generoi raportti
    generator = TestReportGenerator()
    report = generator.generate_report(results)
    
    # Tulosta raportti
    print("\n=== Testiraportti ===")
    print(f"\nYhteenveto:")
    print(f"Testejä yhteensä: {report['summary']['total_tests']}")
    print(f"Onnistuneet: {report['summary']['successful']}")
    print(f"Epäonnistuneet: {report['summary']['failed']}")
    print(f"Onnistumisprosentti: {report['summary']['success_rate']:.1f}%")
    
    print("\nOnnistuneet testit:")
    for test in report["successful_tests"]:
        print(f"- {test['name']} ({test['type']})")
        print(f"  Kesto: {test['duration']:.2f}s")
        print(f"  Details: {test['details']}")
    
    print("\nEpäonnistuneet testit:")
    for test in report["failed_tests"]:
        print(f"- {test['name']} ({test['type']})")
        print(f"  Virhe: {test['error']}")
        print(f"  Kesto: {test['duration']:.2f}s")
        print(f"  Details: {test['details']}")
    
    print("\nParannusehdotukset:")
    for improvement in report["improvements"]:
        print(f"- [{improvement['priority'].upper()}] {improvement['suggestion']}")
        print(f"  Vaikuttaa testeihin: {', '.join(improvement['affected_tests'])}")
        print(f"  Automatisoitavissa: {improvement['automated']}")
        print(f"  Arvioitu työmäärä: {improvement['estimated_effort']}")

if __name__ == "__main__":
    main()
