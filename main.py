#!/usr/bin/env python3
"""
오르다 (Orda) - 투자 학습 플랫폼 백엔드 API
BigKinds 크롤링 + SQLite DB + RAG 분석 + 모의투자 시뮬레이션
"""

import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# 타입 정의
from scripts.types_definition import (
    HealthResponse, AnalysisRequest, AnalysisResponse,
    SimulationRequest, SimulationResponse
)

# 서비스 모듈들
from scripts.crawling_bigkinds import BigKindsCrawler
from scripts.rag_analysis import RAGAnalyzer
from scripts.vector_search import VectorSearchEngine
from scripts.simulation_engine import SimulationEngine
from scripts.database import OrdaDatabase, OrdaDatabaseAPI

# ===== FastAPI 앱 초기화 =====

app = FastAPI(
    title="오르다 API",
    description="투자 학습 플랫폼 백엔드 API - BigKinds 크롤링, RAG 분석, SQLite DB, 모의투자",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
    print(f"✅ static 폴더 연결됨: {static_dir.absolute()}")
else:
    print(f"⚠️ static 폴더가 없습니다: {static_dir.absolute()}")
    static_dir.mkdir(exist_ok=True)

# ===== 전역 서비스 객체들 =====

crawler = None
rag_analyzer = None  
vector_engine = None
simulation_engine = None
db_api = None
orda_db = None

# ===== 서비스 초기화 =====

async def initialize_services():
    """모든 서비스 초기화"""
    global crawler, rag_analyzer, vector_engine, simulation_engine, db_api, orda_db
    
    print("🚀 오르다 서비스 초기화 중...")
    
    try:
        # 1. SQLite 데이터베이스 초기화
        try:
            orda_db = OrdaDatabase()
            db_api = OrdaDatabaseAPI()
            print("✅ SQLite 데이터베이스 API 초기화 완료")
        except Exception as e:
            print(f"⚠️ 데이터베이스 초기화 실패: {e}")
        
        # 2. BigKinds 크롤러 초기화
        try:
            crawler = BigKindsCrawler()
            print("✅ BigKinds 크롤러 초기화 완료")
        except Exception as e:
            print(f"⚠️ BigKinds 크롤러 초기화 실패: {e}")
        
        # 3. 벡터 검색 엔진 초기화
        try:
            vector_engine = VectorSearchEngine()
            await vector_engine.initialize()
            print("✅ 벡터 검색 엔진 초기화 완료")
        except Exception as e:
            print(f"⚠️ 벡터 검색 엔진 초기화 실패: {e}")
        
        # 4. RAG 분석기 초기화
        try:
            rag_analyzer = RAGAnalyzer()
            print("✅ RAG 분석기 초기화 완료")
        except Exception as e:
            print(f"⚠️ RAG 분석기 초기화 실패: {e}")
        
        # 5. 시뮬레이션 엔진 초기화
        try:
            simulation_engine = SimulationEngine()
            print("✅ 시뮬레이션 엔진 초기화 완료")
        except Exception as e:
            print(f"⚠️ 시뮬레이션 엔진 초기화 실패: {e}")
        
        print("🎉 모든 서비스 초기화 완료!")
        
    except Exception as e:
        print(f"❌ 서비스 초기화 실패: {e}")
        print("⚠️ 일부 기능이 제한될 수 있습니다.")

# ===== 메인 라우트 =====

@app.get("/")
async def home():
    """메인 페이지 - index.html로 리다이렉트"""
    return RedirectResponse(url="/static/index.html")

# ===== API 엔드포인트 =====

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """API 서버 및 서비스 상태 확인"""
    components = {}
    
    # SQLite 데이터베이스 상태
    if db_api and orda_db:
        try:
            stats = orda_db.get_database_stats()
            components["sqlite_database"] = {
                "status": "healthy", 
                "description": "SQLite 데이터베이스",
                "details": {
                    "industries": stats.get("industries", 0),
                    "past_issues": stats.get("past_issues", 0),
                    "current_issues": stats.get("current_issues", 0),
                    "db_size_mb": stats.get("db_size_mb", 0)
                }
            }
        except Exception:
            components["sqlite_database"] = {"status": "error", "description": "SQLite 데이터베이스"}
    else:
        components["sqlite_database"] = {"status": "unavailable", "description": "SQLite 데이터베이스"}
    
    # BigKinds 크롤러 상태
    if crawler:
        components["bigkinds_crawler"] = {"status": "healthy", "description": "BigKinds 크롤러"}
    else:
        components["bigkinds_crawler"] = {"status": "unavailable", "description": "BigKinds 크롤러"}
    
    # RAG 분석기 상태
    if rag_analyzer:
        components["rag_analyzer"] = {"status": "healthy", "description": "RAG 분석 시스템"}
    else:
        components["rag_analyzer"] = {"status": "unavailable", "description": "RAG 분석 시스템"}
    
    # 벡터 엔진 상태
    if vector_engine:
        try:
            vector_health = await vector_engine.health_check()
            components["vector_engine"] = {
                "status": vector_health["overall_status"],
                "description": "벡터 검색 엔진",
                "details": vector_health["components"]
            }
        except Exception:
            components["vector_engine"] = {"status": "error", "description": "벡터 검색 엔진"}
    else:
        components["vector_engine"] = {"status": "unavailable", "description": "벡터 검색 엔진"}
    
    # 시뮬레이션 엔진 상태
    if simulation_engine:
        components["simulation_engine"] = {"status": "healthy", "description": "모의투자 시뮬레이션"}
    else:
        components["simulation_engine"] = {"status": "unavailable", "description": "모의투자 시뮬레이션"}
    
    # 전체 상태 결정
    healthy_count = sum(1 for comp in components.values() if comp.get("status") == "healthy")
    total_count = len(components)
    
    if healthy_count == total_count:
        overall_status = "healthy"
    elif healthy_count > 0:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        components=components,
        version="1.0.0"
    )

# ===== BigKinds 크롤링 API =====

@app.get("/api/today-issues")
async def get_today_issues():
    """오늘의 이슈 10개 가져오기 + DB 저장"""
    if not crawler:
        raise HTTPException(status_code=503, detail="BigKinds 크롤러가 초기화되지 않았습니다.")
    
    try:
        print("📡 오늘의 이슈 요청 처리 중...")
        
        # 최신 크롤링 데이터 로드
        latest_data = crawler.load_latest_issues()
        
        if not latest_data:
            print("📡 실시간 BigKinds 크롤링 시작...")
            latest_data = await crawler.crawl_current_issues()
        
        # SQLite 데이터베이스에 저장
        if db_api and latest_data.get("issues"):
            try:
                await db_api.save_current_issues(latest_data["issues"])
                print("✅ 현재 이슈 DB 저장 완료")
            except Exception as save_error:
                print(f"⚠️ 현재 이슈 저장 실패: {save_error}")
        
        return {
            "success": True,
            "data": latest_data["issues"],
            "metadata": {
                "total_issues": latest_data["total_issues"],
                "crawled_at": latest_data["crawled_at"],
                "source": latest_data["source"],
                "saved_to_db": db_api is not None
            }
        }
        
    except Exception as e:
        print(f"❌ 이슈 로드 실패: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"이슈 로드 실패: {str(e)}")

@app.post("/api/refresh-data")
async def refresh_data(background_tasks: BackgroundTasks):
    """데이터 새로고침 (백그라운드 작업)"""
    if not crawler:
        raise HTTPException(status_code=503, detail="크롤러가 초기화되지 않았습니다.")
    
    try:
        background_tasks.add_task(crawler.crawl_and_update)
        return {"success": True, "message": "데이터 새로고침이 백그라운드에서 시작되었습니다."}
        
    except Exception as e:
        print(f"❌ 데이터 새로고침 실패: {e}")
        raise HTTPException(status_code=500, detail=f"데이터 새로고침 실패: {str(e)}")

# ===== SQLite 데이터베이스 API =====

@app.get("/api/past-news")
async def get_past_news_from_db(
    limit: int = 20, 
    search: Optional[str] = None,
    industry: Optional[str] = None
):
    """SQLite에서 과거 뉴스 조회"""
    if not db_api:
        raise HTTPException(status_code=503, detail="데이터베이스가 초기화되지 않았습니다.")
    
    try:
        news_data = await db_api.get_past_news(
            limit=limit, 
            search=search, 
            industry=industry
        )
        
        return {
            "success": True, 
            "data": news_data, 
            "total": len(news_data),
            "source": "sqlite_db",
            "filters": {
                "search": search,
                "industry": industry,
                "limit": limit
            }
        }
        
    except Exception as e:
        print(f"❌ 과거 뉴스 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"과거 뉴스 조회 실패: {str(e)}")

@app.get("/api/industries-db")
async def get_industries_from_db(
    search: Optional[str] = None,
    limit: int = 50
):
    """SQLite에서 산업 분류 조회"""
    if not db_api:
        raise HTTPException(status_code=503, detail="데이터베이스가 초기화되지 않았습니다.")
    
    try:
        industries = await db_api.get_industries(search=search, limit=limit)
        
        return {
            "success": True,
            "data": industries,
            "total": len(industries),
            "source": "sqlite_db"
        }
        
    except Exception as e:
        print(f"❌ 산업 분류 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"산업 분류 조회 실패: {str(e)}")

@app.get("/api/database-stats")
async def get_database_stats():
    """데이터베이스 통계 정보"""
    if not orda_db:
        raise HTTPException(status_code=503, detail="데이터베이스가 초기화되지 않았습니다.")
    
    try:
        stats = orda_db.get_database_stats()
        
        return {
            "success": True,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ 데이터베이스 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")

# ===== 뉴스 분석 API =====

@app.post("/api/analyze-news", response_model=AnalysisResponse)
async def analyze_news(request: AnalysisRequest):
    """뉴스 분석 - 과거 이슈 매칭 + 산업 연결 + RAG 분석"""
    if not all([vector_engine, rag_analyzer]):
        raise HTTPException(status_code=503, detail="분석 서비스가 초기화되지 않았습니다.")
    
    try:
        print(f"📊 뉴스 분석 시작: {request.news_content[:50]}...")
        
        # 1. 과거 유사 이슈 검색 (Pinecone)
        similar_issues = await vector_engine.search_similar_past_issues(
            query=request.news_content,
            top_k=3
        )
        
        # 2. 관련 산업 검색 (Pinecone) 
        related_industries = await vector_engine.search_related_industries(
            query=request.news_content,
            top_k=3
        )
        
        # 3. RAG 종합 분석
        analysis_result = await rag_analyzer.comprehensive_analysis(
            current_news=request.news_content,
            past_issues=similar_issues,
            industries=related_industries
        )
        
        return AnalysisResponse(
            current_news=request.news_content,
            similar_past_issues=similar_issues,
            related_industries=related_industries,
            analysis_result=analysis_result["explanation"],
            confidence_score=analysis_result["confidence"]
        )
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"분석 실패: {str(e)}")

# ===== 섹터 및 기업 정보 API =====

@app.get("/api/sectors")
async def get_sectors():
    """섹터 정보 가져오기 (SQLite + 하드코딩 혼합)"""
    try:
        # SQLite에서 산업 정보 가져오기
        if db_api:
            try:
                db_industries = await db_api.get_industries(limit=20)
                print(f"📊 SQLite에서 {len(db_industries)}개 산업 로드")
            except Exception as e:
                print(f"⚠️ SQLite 산업 조회 실패: {e}")
                db_industries = []
        else:
            db_industries = []
        
        # 하드코딩된 섹터 데이터 (주가, 수익률 등 실시간 정보)
        sectors_data = [
            {
                "id": "semiconductors",
                "name": "반도체",
                "companies": 45,
                "recent_issues": 12,
                "monthly_return": "+5.2%",
                "description": "메모리, 시스템반도체, 반도체 장비"
            },
            {
                "id": "tech",
                "name": "IT 서비스", 
                "companies": 245,
                "recent_issues": 3,
                "monthly_return": "+2.5%",
                "description": "컴퓨터 소프트웨어 개발, 시스템 통합 서비스"
            },
            {
                "id": "finance",
                "name": "금융업",
                "companies": 89,
                "recent_issues": 5, 
                "monthly_return": "+1.2%",
                "description": "은행, 증권, 보험, 자산운용 등 금융 서비스"
            },
            {
                "id": "energy",
                "name": "정유",
                "companies": 34,
                "recent_issues": 8,
                "monthly_return": "+4.1%",
                "description": "원유 정제, 석유화학 제품 생산"
            },
            {
                "id": "defense",
                "name": "방위산업",
                "companies": 23,
                "recent_issues": 7,
                "monthly_return": "+6.3%",
                "description": "방위산업 관련 장비, 무기체계, 항공우주"
            },
            {
                "id": "healthcare",
                "name": "의료·정밀기기",
                "companies": 156,
                "recent_issues": 4,
                "monthly_return": "+3.8%",
                "description": "의료기기, 정밀기기, 바이오 관련 제품"
            },
            {
                "id": "chemicals",
                "name": "화학",
                "companies": 98,
                "recent_issues": 6,
                "monthly_return": "+2.9%",
                "description": "기초 화학물질, 정밀화학, 플라스틱"
            },
            {
                "id": "automotive",
                "name": "운송장비·부품",
                "companies": 87,
                "recent_issues": 5,
                "monthly_return": "+1.7%",
                "description": "자동차, 조선, 항공기 등 운송장비 및 부품"
            },
            {
                "id": "consumer",
                "name": "음식료·담배",
                "companies": 67,
                "recent_issues": 2,
                "monthly_return": "+0.8%",
                "description": "식품, 음료, 담배 등 소비재 제조"
            }
        ]
        
        return {
            "success": True, 
            "data": sectors_data,
            "db_industries_count": len(db_industries),
            "data_source": "mixed_sqlite_hardcoded"
        }
        
    except Exception as e:
        print(f"❌ 섹터 데이터 로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"섹터 데이터 로드 실패: {str(e)}")

@app.get("/api/companies")
async def get_companies(sector: Optional[str] = None):
    """기업 정보 가져오기 (섹터별 필터링 가능)"""
    try:
        # 기업 데이터 (실제 시세 반영)
        companies_data = [
            {
                "id": 1,
                "name": "삼성전자",
                "code": "005930", 
                "sector": "반도체",
                "market": "KOSPI",
                "price": 74800,
                "change": "+1200",
                "change_rate": "+1.63%",
                "market_cap": "447조원",
                "per": "15.2",
                "pbr": "1.4"
            },
            {
                "id": 2,
                "name": "SK하이닉스",
                "code": "000660",
                "sector": "반도체",
                "market": "KOSPI",
                "price": 89600,
                "change": "-800",
                "change_rate": "-0.88%",
                "market_cap": "65조원",
                "per": "12.8",
                "pbr": "1.1"
            },
            {
                "id": 3,
                "name": "S-OIL",
                "code": "010950",
                "sector": "정유",
                "market": "KOSPI",
                "price": 68900,
                "change": "+2100",
                "change_rate": "+3.14%",
                "market_cap": "8.2조원",
                "per": "8.5",
                "pbr": "0.9"
            },
            {
                "id": 4,
                "name": "한국항공우주",
                "code": "047810",
                "sector": "방위산업",
                "market": "KOSPI",
                "price": 42350,
                "change": "+1850",
                "change_rate": "+4.57%",
                "market_cap": "1.9조원",
                "per": "18.2",
                "pbr": "2.1"
            },
            {
                "id": 5,
                "name": "LG화학",
                "code": "051910",
                "sector": "화학",
                "market": "KOSPI",
                "price": 385000,
                "change": "-5000",
                "change_rate": "-1.28%",
                "market_cap": "27조원",
                "per": "22.1",
                "pbr": "1.8"
            },
            {
                "id": 6,
                "name": "카카오",
                "code": "035720",
                "sector": "IT 서비스",
                "market": "KOSPI",
                "price": 58400,
                "change": "+900",
                "change_rate": "+1.56%",
                "market_cap": "25조원",
                "per": "N/A",
                "pbr": "2.3"
            },
            {
                "id": 7,
                "name": "네이버",
                "code": "035420",
                "sector": "IT 서비스",
                "market": "KOSPI",
                "price": 178500,
                "change": "-2500",
                "change_rate": "-1.38%",
                "market_cap": "29조원",
                "per": "16.8",
                "pbr": "1.9"
            },
            {
                "id": 8,
                "name": "KB금융",
                "code": "105560",
                "sector": "금융업",
                "market": "KOSPI",
                "price": 68200,
                "change": "+1100",
                "change_rate": "+1.64%",
                "market_cap": "28조원",
                "per": "6.2",
                "pbr": "0.5"
            }
        ]
        
        # 섹터 필터링
        if sector:
            companies_data = [c for c in companies_data if c["sector"] == sector]
            print(f"🏭 '{sector}' 섹터 필터링: {len(companies_data)}개 기업")
        
        return {
            "success": True, 
            "data": companies_data,
            "filtered_by_sector": sector,
            "total_companies": len(companies_data)
        }
        
    except Exception as e:
        print(f"❌ 기업 데이터 로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"기업 데이터 로드 실패: {str(e)}")

# ===== 모의투자 시뮬레이션 API =====

@app.get("/api/scenarios")
async def get_simulation_scenarios():
    """모의투자 시뮬레이션 시나리오 목록"""
    if not simulation_engine:
        raise HTTPException(status_code=503, detail="시뮬레이션 엔진이 초기화되지 않았습니다.")
    
    try:
        scenarios = simulation_engine.get_available_scenarios()
        return {"success": True, "data": scenarios}
        
    except Exception as e:
        print(f"❌ 시나리오 로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"시나리오 로드 실패: {str(e)}")

@app.get("/api/scenarios/{scenario_id}")
async def get_scenario_detail(scenario_id: str):
    """특정 시나리오 상세 정보"""
    if not simulation_engine:
        raise HTTPException(status_code=503, detail="시뮬레이션 엔진이 초기화되지 않았습니다.")
    
    try:
        scenario_info = await simulation_engine.get_scenario_info(scenario_id)
        if not scenario_info:
            raise HTTPException(status_code=404, detail=f"시나리오를 찾을 수 없습니다: {scenario_id}")
        
        # 추천 종목도 함께 반환
        recommended_stocks = simulation_engine.get_recommended_stocks_for_scenario(scenario_id)
        
        return {
            "success": True,
            "scenario": scenario_info,
            "recommended_stocks": recommended_stocks
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 시나리오 상세 정보 로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"시나리오 정보 로드 실패: {str(e)}")

@app.post("/api/mock-invest", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    """모의투자 시뮬레이션 실행 + DB 저장"""
    if not simulation_engine:
        raise HTTPException(status_code=503, detail="시뮬레이션 엔진이 초기화되지 않았습니다.")
    
    try:
        print(f"🎯 시뮬레이션 요청: {request.scenario_id}, {request.investment_amount:,}원")
        
        # 입력값 검증
        validation = await simulation_engine.validate_simulation_inputs(
            scenario_id=request.scenario_id,
            investment_amount=request.investment_amount,
            investment_period=request.investment_period,
            selected_stocks=request.selected_stocks
        )
        
        if not validation["valid"]:
            raise HTTPException(
                status_code=400, 
                detail={
                    "message": "입력값 검증 실패",
                    "errors": validation["errors"],
                    "warnings": validation["warnings"]
                }
            )
        
        # 시뮬레이션 실행
        result = await simulation_engine.run_simulation(
            scenario_id=request.scenario_id,
            investment_amount=request.investment_amount,
            investment_period=request.investment_period,
            selected_stocks=request.selected_stocks
        )
        
        # SQLite 데이터베이스에 결과 저장
        if db_api:
            try:
                user_session = "demo_user"  # 실제로는 IP나 세션 ID 사용
                await db_api.save_simulation_result(
                    scenario_id=request.scenario_id,
                    user_session=user_session,
                    investment_amount=request.investment_amount,
                    investment_period=request.investment_period,
                    selected_stocks=request.selected_stocks,
                    simulation_result=result
                )
                print("✅ 시뮬레이션 결과 DB 저장 완료")
            except Exception as save_error:
                print(f"⚠️ 시뮬레이션 결과 저장 실패: {save_error}")
        
        return SimulationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 시뮬레이션 실패: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"시뮬레이션 실패: {str(e)}")

@app.post("/api/validate-simulation")
async def validate_simulation_inputs(request: SimulationRequest):
    """시뮬레이션 입력값 검증"""
    if not simulation_engine:
        raise HTTPException(status_code=503, detail="시뮬레이션 엔진이 초기화되지 않았습니다.")
    
    try:
        validation = await simulation_engine.validate_simulation_inputs(
            scenario_id=request.scenario_id,
            investment_amount=request.investment_amount,
            investment_period=request.investment_period,
            selected_stocks=request.selected_stocks
        )
        
        return {"success": True, "validation": validation}
        
    except Exception as e:
        print(f"❌ 입력값 검증 실패: {e}")
        raise HTTPException(status_code=500, detail=f"입력값 검증 실패: {str(e)}")

# ===== 벡터 및 통계 API =====

@app.get("/api/vector-stats")
async def get_vector_stats():
    """벡터 데이터베이스 통계"""
    if not vector_engine:
        raise HTTPException(status_code=503, detail="벡터 엔진이 초기화되지 않았습니다.")
    
    try:
        stats = vector_engine.get_index_stats()
        return {"success": True, "stats": stats}
        
    except Exception as e:
        print(f"❌ 벡터 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"벡터 통계 조회 실패: {str(e)}")

# ===== 추가 유틸리티 API =====

@app.get("/api/search-companies")
async def search_companies(query: str, limit: int = 10):
    """기업 검색 (이름, 코드 기준)"""
    try:
        # 모든 기업 데이터에서 검색
        all_companies = [
            {"id": 1, "name": "삼성전자", "code": "005930", "sector": "반도체"},
            {"id": 2, "name": "SK하이닉스", "code": "000660", "sector": "반도체"},
            {"id": 3, "name": "S-OIL", "code": "010950", "sector": "정유"},
            {"id": 4, "name": "한국항공우주", "code": "047810", "sector": "방위산업"},
            {"id": 5, "name": "LG화학", "code": "051910", "sector": "화학"},
            {"id": 6, "name": "카카오", "code": "035720", "sector": "IT 서비스"},
            {"id": 7, "name": "네이버", "code": "035420", "sector": "IT 서비스"},
            {"id": 8, "name": "KB금융", "code": "105560", "sector": "금융업"},
            # 추가 기업들...
            {"id": 9, "name": "현대자동차", "code": "005380", "sector": "운송장비·부품"},
            {"id": 10, "name": "POSCO홀딩스", "code": "005490", "sector": "철강"},
            {"id": 11, "name": "셀트리온", "code": "068270", "sector": "의료·정밀기기"},
            {"id": 12, "name": "LG에너지솔루션", "code": "373220", "sector": "배터리"},
        ]
        
        # 검색 실행
        query_lower = query.lower()
        filtered_companies = [
            company for company in all_companies
            if query_lower in company["name"].lower() or 
               query_lower in company["code"].lower() or
               query_lower in company["sector"].lower()
        ]
        
        # 결과 제한
        result = filtered_companies[:limit]
        
        return {
            "success": True,
            "data": result,
            "query": query,
            "total_found": len(filtered_companies),
            "returned": len(result)
        }
        
    except Exception as e:
        print(f"❌ 기업 검색 실패: {e}")
        raise HTTPException(status_code=500, detail=f"기업 검색 실패: {str(e)}")

@app.get("/api/simulation-history")
async def get_simulation_history(limit: int = 20):
    """시뮬레이션 실행 이력 (SQLite에서 조회)"""
    if not db_api:
        raise HTTPException(status_code=503, detail="데이터베이스가 초기화되지 않았습니다.")
    
    try:
        # SQLite에서 시뮬레이션 이력 조회
        import aiosqlite
        
        async with aiosqlite.connect(db_api.db_path) as db:
            query = """
                SELECT scenario_id, investment_amount, investment_period, 
                       total_return_pct, final_amount, created_at
                FROM simulation_results 
                ORDER BY created_at DESC 
                LIMIT ?
            """
            
            async with db.execute(query, (limit,)) as cursor:
                rows = await cursor.fetchall()
                
                history = []
                for row in rows:
                    history.append({
                        "scenario_id": row[0],
                        "investment_amount": row[1],
                        "investment_period": row[2],
                        "total_return_pct": row[3],
                        "final_amount": row[4],
                        "created_at": row[5]
                    })
                
                return {
                    "success": True,
                    "data": history,
                    "total": len(history)
                }
        
    except Exception as e:
        print(f"❌ 시뮬레이션 이력 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"시뮬레이션 이력 조회 실패: {str(e)}")

@app.get("/api/trending-issues")
async def get_trending_issues(days: int = 7):
    """최근 며칠간 인기 이슈 (현재 이슈 테이블 기준)"""
    if not db_api:
        # DB가 없으면 하드코딩된 트렌딩 이슈 반환
        trending_data = [
            {
                "rank": 1,
                "title": "반도체 수출 급증으로 국가 경제 긍정 신호",
                "mentions": 156,
                "sentiment": "positive",
                "related_sectors": ["반도체", "IT 서비스"]
            },
            {
                "rank": 2,
                "title": "방위산업 관련 주요 계약 체결 소식",
                "mentions": 134,
                "sentiment": "positive", 
                "related_sectors": ["방위산업"]
            },
            {
                "rank": 3,
                "title": "정유업계 글로벌 유가 변동 대응",
                "mentions": 98,
                "sentiment": "neutral",
                "related_sectors": ["정유", "화학"]
            }
        ]
        
        return {
            "success": True,
            "data": trending_data,
            "source": "hardcoded",
            "period_days": days
        }
    
    try:
        import aiosqlite
        from datetime import datetime, timedelta
        
        async with aiosqlite.connect(db_api.db_path) as db:
            # 최근 N일간 이슈를 그룹화하여 인기 키워드 추출
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            query = """
                SELECT title, COUNT(*) as mentions
                FROM current_issues 
                WHERE created_at >= ?
                GROUP BY LOWER(title)
                ORDER BY mentions DESC
                LIMIT 10
            """
            
            async with db.execute(query, (since_date,)) as cursor:
                rows = await cursor.fetchall()
                
                trending = []
                for idx, row in enumerate(rows, 1):
                    trending.append({
                        "rank": idx,
                        "title": row[0],
                        "mentions": row[1],
                        "sentiment": "neutral",  # TODO: 실제 감정 분석 구현
                        "related_sectors": []    # TODO: 실제 섹터 매핑 구현
                    })
                
                return {
                    "success": True,
                    "data": trending,
                    "source": "sqlite_db",
                    "period_days": days
                }
    
    except Exception as e:
        print(f"❌ 트렌딩 이슈 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"트렌딩 이슈 조회 실패: {str(e)}")

# ===== 서버 시작 시 초기화 =====

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행"""
    print("🎯 오르다 API 서버 시작 중...")
    await initialize_services()
    print("🚀 오르다 API 서버 시작 완료!")

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 실행"""
    print("🔄 오르다 API 서버 종료 중...")
    
    # 리소스 정리
    global crawler, rag_analyzer, vector_engine, simulation_engine, db_api
    
    if vector_engine:
        try:
            await vector_engine.close()
            print("✅ 벡터 엔진 종료 완료")
        except Exception as e:
            print(f"⚠️ 벡터 엔진 종료 실패: {e}")
    
    print("👋 오르다 API 서버 종료 완료!")

# ===== 메인 실행 =====

if __name__ == "__main__":
    import uvicorn
    
    print("🎯 오르다 투자 학습 플랫폼 API 서버")
    print("=" * 50)
    print("📊 BigKinds 크롤링 + SQLite DB + RAG 분석 + 모의투자")
    print("🌐 http://localhost:8000")
    print("📋 API 문서: http://localhost:8000/docs")
    print("🏠 프론트엔드: http://localhost:8000/static/index.html")
    print("=" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )