#!/usr/bin/env python3
"""
SQLite 데이터베이스 구축 및 관리 시스템
CSV 데이터를 SQLite DB로 변환하고 관리
"""

import sqlite3
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio
import aiosqlite

class OrdaDatabase:
    """오르다 프로젝트 SQLite 데이터베이스 관리 클래스"""
    
    def __init__(self, db_path: str = "data/orda.db"):
        """
        초기화
        
        Args:
            db_path: SQLite 데이터베이스 파일 경로
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        print(f"🗄️ SQLite 데이터베이스 초기화: {self.db_path}")
    
    def create_tables(self):
        """데이터베이스 테이블 생성"""
        
        create_tables_sql = """
        -- 산업 분류 테이블
        CREATE TABLE IF NOT EXISTS industries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            krx_name TEXT NOT NULL UNIQUE,          -- KRX 업종명
            description TEXT,                       -- 상세내용
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 과거 이슈 테이블
        CREATE TABLE IF NOT EXISTS past_issues (
            id TEXT PRIMARY KEY,                    -- ID (CSV의 ID 컬럼)
            issue_name TEXT NOT NULL,               -- Issue_name
            contents TEXT,                          -- Contents
            contents_spec TEXT,                     -- Contentes(Spec)
            related_industries TEXT,                -- 관련 산업
            industry_reason TEXT,                   -- 산업 이유
            start_date TEXT,                        -- Start_date
            end_date TEXT,                          -- Fin_date
            evidence_source TEXT,                   -- 근거자료
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 현재 이슈 테이블 (BigKinds 크롤링)
        CREATE TABLE IF NOT EXISTS current_issues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue_number INTEGER,                   -- 이슈번호
            title TEXT NOT NULL,                    -- 제목
            content TEXT,                           -- 내용
            crawled_at TIMESTAMP,                   -- 추출시간
            source TEXT DEFAULT 'bigkinds',        -- 출처
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 시뮬레이션 결과 테이블
        CREATE TABLE IF NOT EXISTS simulation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario_id TEXT NOT NULL,              -- 시나리오 ID
            user_session TEXT,                      -- 사용자 세션 (IP 등)
            investment_amount INTEGER,              -- 투자 금액
            investment_period INTEGER,              -- 투자 기간 (개월)
            selected_stocks TEXT,                   -- 선택 종목 (JSON)
            total_return_pct REAL,                 -- 총 수익률
            final_amount INTEGER,                   -- 최종 금액
            simulation_data TEXT,                   -- 전체 시뮬레이션 결과 (JSON)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 인덱스 생성
        CREATE INDEX IF NOT EXISTS idx_industries_krx_name ON industries(krx_name);
        CREATE INDEX IF NOT EXISTS idx_past_issues_related_industries ON past_issues(related_industries);
        CREATE INDEX IF NOT EXISTS idx_past_issues_start_date ON past_issues(start_date);
        CREATE INDEX IF NOT EXISTS idx_current_issues_crawled_at ON current_issues(crawled_at);
        CREATE INDEX IF NOT EXISTS idx_simulation_results_scenario_id ON simulation_results(scenario_id);
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(create_tables_sql)
                print("✅ 데이터베이스 테이블 생성 완료")
                
        except Exception as e:
            print(f"❌ 테이블 생성 실패: {e}")
            raise
    
    def import_csv_data(self, csv_dir: str = "data"):
        """CSV 파일들을 SQLite로 가져오기"""
        csv_path = Path(csv_dir)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                
                # 1. 산업 분류 데이터 가져오기
                industry_csv = csv_path / "산업DB.v.0.3.csv"
                if industry_csv.exists():
                    print(f"📊 산업 분류 데이터 가져오기: {industry_csv}")
                    
                    df = pd.read_csv(industry_csv)
                    
                    # 데이터 정리
                    df = df.dropna(subset=['KRX 업종명'])  # 필수 컬럼 체크
                    df['상세내용'] = df['상세내용'].fillna('')  # 빈 값 처리
                    
                    # 중복 제거
                    df = df.drop_duplicates(subset=['KRX 업종명'])
                    
                    # SQLite에 삽입
                    for _, row in df.iterrows():
                        conn.execute("""
                            INSERT OR REPLACE INTO industries (krx_name, description)
                            VALUES (?, ?)
                        """, (row['KRX 업종명'], row['상세내용']))
                    
                    print(f"✅ 산업 분류 {len(df)}건 가져오기 완료")
                
                # 2. 과거 이슈 데이터 가져오기
                past_news_csv = csv_path / "Past_news.csv"
                if past_news_csv.exists():
                    print(f"📰 과거 이슈 데이터 가져오기: {past_news_csv}")
                    
                    df = pd.read_csv(past_news_csv)
                    
                    # 데이터 정리
                    df = df.dropna(subset=['ID'])  # 필수 컬럼 체크
                    df = df.fillna('')  # 빈 값 처리
                    
                    # SQLite에 삽입
                    for _, row in df.iterrows():
                        conn.execute("""
                            INSERT OR REPLACE INTO past_issues 
                            (id, issue_name, contents, contents_spec, related_industries, 
                             industry_reason, start_date, end_date, evidence_source)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            row['ID'], 
                            row['Issue_name'], 
                            row['Contents'],
                            row.get('Contentes(Spec)', ''),  # 오타 있는 컬럼명 처리
                            row['관련 산업'],
                            row['산업 이유'],
                            row['Start_date'],
                            row['Fin_date'],
                            row['근거자료']
                        ))
                    
                    print(f"✅ 과거 이슈 {len(df)}건 가져오기 완료")
                
                conn.commit()
                print("🎉 모든 CSV 데이터 가져오기 완료")
                
        except Exception as e:
            print(f"❌ CSV 데이터 가져오기 실패: {e}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """데이터베이스 통계 정보"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # 각 테이블별 데이터 수
                tables = ['industries', 'past_issues', 'current_issues', 'simulation_results']
                
                for table in tables:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    stats[table] = count
                
                # 데이터베이스 파일 크기
                stats['db_size_mb'] = round(self.db_path.stat().st_size / (1024 * 1024), 2)
                stats['db_path'] = str(self.db_path)
                stats['last_updated'] = datetime.now().isoformat()
                
                return stats
                
        except Exception as e:
            print(f"❌ 통계 조회 실패: {e}")
            return {}

class OrdaDatabaseAPI:
    """오르다 데이터베이스 API 클래스 (비동기)"""
    
    def __init__(self, db_path: str = "data/orda.db"):
        self.db_path = db_path
    
    async def get_past_news(
        self, 
        limit: int = 20, 
        search: Optional[str] = None,
        industry: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """과거 뉴스 조회"""
        
        query = """
            SELECT id, issue_name, contents, related_industries, 
                   industry_reason, start_date, end_date, evidence_source
            FROM past_issues 
            WHERE 1=1
        """
        params = []
        
        # 검색 조건 추가
        if search:
            query += " AND (issue_name LIKE ? OR contents LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])
        
        if industry:
            query += " AND related_industries LIKE ?"
            params.append(f"%{industry}%")
        
        query += " ORDER BY start_date DESC LIMIT ?"
        params.append(limit)
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    
                    results = []
                    for row in rows:
                        issue = {
                            "id": row[0],
                            "title": row[1],
                            "summary": row[2][:200] + "..." if len(row[2]) > 200 else row[2],
                            "content": row[2],
                            "related_industries": row[3],
                            "industry_reason": row[4],
                            "start_date": row[5],
                            "end_date": row[6],
                            "source": row[7] or "과거 이슈",
                            "date": row[5],
                            "type": "과거 이슈"
                        }
                        results.append(issue)
                    
                    return results
                    
        except Exception as e:
            print(f"❌ 과거 뉴스 조회 실패: {e}")
            return []
    
    async def get_industries(
        self, 
        search: Optional[str] = None, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """산업 분류 조회"""
        
        query = "SELECT krx_name, description FROM industries WHERE 1=1"
        params = []
        
        if search:
            query += " AND (krx_name LIKE ? OR description LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])
        
        query += " ORDER BY krx_name LIMIT ?"
        params.append(limit)
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    
                    results = []
                    for row in rows:
                        industry = {
                            "name": row[0],
                            "description": row[1],
                            "companies": 0,  # TODO: 실제 기업 수 계산
                            "recent_issues": 0,  # TODO: 관련 이슈 수 계산
                            "monthly_return": "+0.0%"  # TODO: 실제 수익률 연동
                        }
                        results.append(industry)
                    
                    return results
                    
        except Exception as e:
            print(f"❌ 산업 분류 조회 실패: {e}")
            return []
    
    async def save_simulation_result(
        self,
        scenario_id: str,
        user_session: str,
        investment_amount: int,
        investment_period: int,
        selected_stocks: List[Dict],
        simulation_result: Dict[str, Any]
    ) -> bool:
        """시뮬레이션 결과 저장"""
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO simulation_results 
                    (scenario_id, user_session, investment_amount, investment_period, 
                     selected_stocks, total_return_pct, final_amount, simulation_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    scenario_id,
                    user_session,
                    investment_amount,
                    investment_period,
                    json.dumps(selected_stocks, ensure_ascii=False),
                    simulation_result.get('simulation_results', {}).get('total_return_pct', 0),
                    simulation_result.get('simulation_results', {}).get('final_amount', 0),
                    json.dumps(simulation_result, ensure_ascii=False)
                ))
                
                await db.commit()
                return True
                
        except Exception as e:
            print(f"❌ 시뮬레이션 결과 저장 실패: {e}")
            return False
    
    async def save_current_issues(self, issues: List[Dict[str, Any]]) -> bool:
        """현재 이슈 저장 (BigKinds 크롤링 결과)"""
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # 기존 당일 데이터 삭제
                today = datetime.now().strftime('%Y-%m-%d')
                await db.execute(
                    "DELETE FROM current_issues WHERE DATE(crawled_at) = ?", 
                    (today,)
                )
                
                # 새 데이터 삽입
                for issue in issues:
                    await db.execute("""
                        INSERT INTO current_issues 
                        (issue_number, title, content, crawled_at, source)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        issue.get('이슈번호', 0),
                        issue.get('제목', ''),
                        issue.get('내용', ''),
                        issue.get('추출시간', datetime.now().isoformat()),
                        'bigkinds'
                    ))
                
                await db.commit()
                print(f"✅ 현재 이슈 {len(issues)}건 저장 완료")
                return True
                
        except Exception as e:
            print(f"❌ 현재 이슈 저장 실패: {e}")
            return False

# ===== 데이터베이스 초기화 및 설정 함수 =====

def setup_database():
    """데이터베이스 초기 설정 (한 번만 실행)"""
    print("🗄️ 오르다 SQLite 데이터베이스 설정 시작")
    
    try:
        # 1. 데이터베이스 생성
        db = OrdaDatabase()
        
        # 2. 테이블 생성
        db.create_tables()
        
        # 3. CSV 데이터 가져오기
        db.import_csv_data()
        
        # 4. 통계 출력
        stats = db.get_database_stats()
        print("\n📊 데이터베이스 통계:")
        for key, value in stats.items():
            print(f"  • {key}: {value}")
        
        print("\n🎉 데이터베이스 설정 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 데이터베이스 설정 실패: {e}")
        return False

def test_database():
    """데이터베이스 테스트"""
    print("🧪 데이터베이스 테스트 시작")
    
    async def test_queries():
        api = OrdaDatabaseAPI()
        
        # 과거 뉴스 테스트
        print("\n📰 과거 뉴스 조회 테스트:")
        past_news = await api.get_past_news(limit=5)
        for news in past_news:
            print(f"  • {news['title'][:50]}...")
        
        # 산업 분류 테스트
        print("\n🏭 산업 분류 조회 테스트:")
        industries = await api.get_industries(limit=5)
        for industry in industries:
            print(f"  • {industry['name']}")
        
        print("✅ 데이터베이스 테스트 완료")
    
    # 비동기 함수 실행
    asyncio.run(test_queries())

# ===== 직접 실행 시 =====

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "setup":
            setup_database()
        elif command == "test":
            test_database()
        elif command == "stats":
            db = OrdaDatabase()
            stats = db.get_database_stats()
            print("📊 데이터베이스 통계:")
            for key, value in stats.items():
                print(f"  • {key}: {value}")
        else:
            print("사용법: python database.py [setup|test|stats]")
    else:
        print("🗄️ 오르다 SQLite 데이터베이스 관리")
        print("사용법:")
        print("  python database.py setup  # 데이터베이스 초기 설정")
        print("  python database.py test   # 데이터베이스 테스트")
        print("  python database.py stats  # 통계 조회")