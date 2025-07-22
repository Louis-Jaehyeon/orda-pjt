#!/usr/bin/env python3
"""
모의투자 시뮬레이션 엔진 - FastAPI 연동용
과거 이슈 시점 기준 "만약 그때 투자했다면?" 시뮬레이션
"""

import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path
import traceback
from dataclasses import dataclass
from enum import Enum

@dataclass
class SimulationStock:
    """시뮬레이션용 종목 정보"""
    code: str
    name: str
    industry: str
    allocation: float  # 투자 비중 (%)
    amount: int        # 투자 금액 (원)

@dataclass
class SimulationScenario:
    """시뮬레이션 시나리오"""
    id: str
    name: str
    description: str
    start_date: str
    end_date: str
    related_industries: List[str]
    impact_description: str
    expected_sectors: Dict[str, str]  # 섹터별 예상 영향

class SimulationStatus(Enum):
    """시뮬레이션 상태"""
    INITIALIZING = "initializing"
    FETCHING_DATA = "fetching_data"
    CALCULATING = "calculating"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"

class SimulationEngine:
    """모의투자 시뮬레이션 엔진"""
    
    def __init__(self, data_dir: str = "data"):
        """
        초기화
        
        Args:
            data_dir: 시뮬레이션 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 과거 이슈 시나리오 데이터 (DB 과거이슈 Table 기반)
        self.scenarios = self._load_scenarios()
        
        # 종목 코드 매핑 (한국 → 야후 파이낸스)
        self.stock_code_mapping = self._load_stock_mapping()
        
        # 거시경제 지표 (코스피, 환율 등)
        self.market_indices = {
            "KOSPI": "^KS11",
            "KOSDAQ": "^KQ11", 
            "USD_KRW": "KRW=X",
            "WTI": "CL=F"
        }
    
    def _load_scenarios(self) -> Dict[str, SimulationScenario]:
        """과거 이슈 시나리오 로드"""
        scenarios = {
            "PN_001": SimulationScenario(
                id="PN_001",
                name="미국 정권 교체와 초대형 부양책 발표",
                description="바이든 행정부 출범과 함께 1.9조 달러 구제법안과 2.7조 달러 인프라 투자",
                start_date="2021-01-01",
                end_date="2021-04-30",
                related_industries=["건설", "화학", "금속", "친환경"],
                impact_description="대규모 인프라 투자로 건설, 철강, 시멘트 등 기초소재 산업 수혜",
                expected_sectors={
                    "건설": "인프라 투자 직접 수혜",
                    "화학": "친환경 정책 관련 수혜",
                    "금속": "철강 수요 증가"
                }
            ),
            "PN_002": SimulationScenario(
                id="PN_002", 
                name="글로벌 금리 상승과 기술주 조정",
                description="미국 10년물 국채 금리 1.7% 돌파로 고평가 성장주 조정",
                start_date="2021-02-01",
                end_date="2021-05-31",
                related_industries=["IT 서비스", "제약", "바이오"],
                impact_description="금리 상승으로 고PER 기술주, 바이오주 대폭 조정",
                expected_sectors={
                    "IT 서비스": "고평가 조정",
                    "제약": "미래가치 할인율 상승 부담"
                }
            ),
            "PN_003": SimulationScenario(
                id="PN_003",
                name="탄소중립 및 신재생에너지 정책 확대", 
                description="한국 정부 2050 탄소중립 선언과 구체적 이행 로드맵 발표",
                start_date="2021-03-01",
                end_date="2021-08-31",
                related_industries=["전기전자", "화학", "운송장비"],
                impact_description="전기차, 2차전지, 태양광 등 친환경 산업 급성장",
                expected_sectors={
                    "전기전자": "2차전지, 전기차 부품 수혜",
                    "화학": "배터리 소재 수혜",
                    "운송장비": "전기차 전환 가속화"
                }
            ),
            "PN_004": SimulationScenario(
                id="PN_004",
                name="코로나19 재확산과 델타·오미크론 등장",
                description="변이 바이러스로 리오프닝 지연, 비대면 산업 재조명",
                start_date="2021-07-01", 
                end_date="2021-12-31",
                related_industries=["운송·창고", "의료·정밀기기", "IT 서비스"],
                impact_description="비대면 서비스 재부상, 진단키트 업체 호황",
                expected_sectors={
                    "의료·정밀기기": "진단키트, 의료기기 수요 급증",
                    "IT 서비스": "원격근무, 비대면 서비스 수혜"
                }
            ),
            "PN_005": SimulationScenario(
                id="PN_005",
                name="이란 솔레이마니 제거 사건",
                description="미군의 이란 쿠드스 부대 사령관 제거로 중동 긴장 극도화",
                start_date="2020-01-01",
                end_date="2020-04-30", 
                related_industries=["정유", "방위산업", "금융"],
                impact_description="지정학적 리스크로 유가 급등, 방산주 강세",
                expected_sectors={
                    "정유": "유가 상승 직접 수혜",
                    "방위산업": "지정학적 리스크 수혜",
                    "금융": "피한 자산 선호"
                }
            ),
            "PN_006": SimulationScenario(
                id="PN_006",
                name="일본 반도체 소재 수출 규제",
                description="일본의 반도체 핵심 소재 수출 규제로 국내 반도체 업계 타격",
                start_date="2019-07-01",
                end_date="2020-01-31",
                related_industries=["반도체", "화학", "IT 서비스"],
                impact_description="초기 타격 후 대체재 개발로 반전 성공",
                expected_sectors={
                    "반도체": "초기 타격 후 대체재 개발",
                    "화학": "소재 국산화 수혜"
                }
            )
        }
        return scenarios
    
    def _load_stock_mapping(self) -> Dict[str, str]:
        """한국 종목코드 → 야후 파이낸스 코드 매핑"""
        return {
            # IT 서비스
            "035720": "035720.KS",  # 카카오
            "035420": "035420.KS",  # 네이버
            "053800": "053800.KQ",  # 안랩
            
            # 반도체
            "005930": "005930.KS",  # 삼성전자
            "000660": "000660.KS",  # SK하이닉스
            "042700": "042700.KQ",  # 한미반도체
            
            # 정유
            "010950": "010950.KS",  # S-OIL
            "078930": "078930.KS",  # GS
            "096770": "096770.KS",  # SK이노베이션
            
            # 방위산업
            "047810": "047810.KS",  # 한국항공우주
            "079550": "079550.KQ",  # LIG넥스원
            "012450": "012450.KS",  # 한화시스템
            
            # 화학
            "051910": "051910.KS",  # LG화학
            "000270": "000270.KS",  # 기아
            
            # 의료·정밀기기
            "145020": "145020.KQ",  # 휴젤
            "196170": "196170.KQ",  # 알테오젠
            "182400": "182400.KQ",  # 엔케이맥스
            
            # 금융
            "105560": "105560.KS",  # KB금융
            
            # 시장 지수
            "KOSPI": "^KS11",
            "KOSDAQ": "^KQ11"
        }
    
    async def run_simulation(
        self,
        scenario_id: str,
        investment_amount: int,
        investment_period: int,  # months
        selected_stocks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        모의투자 시뮬레이션 실행
        
        Args:
            scenario_id: 시나리오 ID
            investment_amount: 투자 금액
            investment_period: 투자 기간 (개월)
            selected_stocks: 선택된 종목 리스트
            
        Returns:
            시뮬레이션 결과
        """
        try:
            print(f"🎯 모의투자 시뮬레이션 시작: {scenario_id}")
            
            # 1. 시나리오 정보 로드
            scenario = self.scenarios.get(scenario_id)
            if not scenario:
                raise ValueError(f"시나리오를 찾을 수 없습니다: {scenario_id}")
            
            # 2. 투자 종목 정보 구성
            simulation_stocks = self._prepare_simulation_stocks(
                selected_stocks, investment_amount
            )
            
            # 3. 시뮬레이션 기간 계산
            sim_start_date = pd.to_datetime(scenario.start_date)
            sim_end_date = min(
                pd.to_datetime(scenario.end_date),
                sim_start_date + pd.DateOffset(months=investment_period)
            )
            
            print(f"📅 시뮬레이션 기간: {sim_start_date.date()} ~ {sim_end_date.date()}")
            
            # 4. 주가 데이터 수집
            stock_data = await self._fetch_stock_data(simulation_stocks, sim_start_date, sim_end_date)
            
            # 5. 시장 지수 데이터 수집
            market_data = await self._fetch_market_data(sim_start_date, sim_end_date)
            
            # 6. 포트폴리오 성과 계산
            portfolio_performance = self._calculate_portfolio_performance(
                simulation_stocks, stock_data, sim_start_date, sim_end_date
            )
            
            # 7. 종목별 성과 분석
            stock_analysis = self._analyze_individual_stocks(
                simulation_stocks, stock_data, sim_start_date, sim_end_date
            )
            
            # 8. 시장 대비 성과 분석
            market_comparison = self._compare_with_market(
                portfolio_performance, market_data, sim_start_date, sim_end_date
            )
            
            # 9. 학습 포인트 생성
            learning_points = self._generate_learning_points(
                scenario, portfolio_performance, stock_analysis, market_comparison
            )
            
            # 10. 결과 종합
            result = {
                "scenario_info": {
                    "id": scenario.id,
                    "name": scenario.name,
                    "description": scenario.description,
                    "period": f"{sim_start_date.strftime('%Y.%m')} - {sim_end_date.strftime('%Y.%m')}",
                    "investment_period_months": investment_period
                },
                "simulation_results": {
                    "initial_amount": investment_amount,
                    "final_amount": portfolio_performance["final_value"],
                    "total_return": portfolio_performance["total_return"],
                    "total_return_pct": portfolio_performance["total_return_pct"],
                    "daily_returns": portfolio_performance["daily_returns"],
                    "volatility": portfolio_performance["volatility"],
                    "sharpe_ratio": portfolio_performance["sharpe_ratio"]
                },
                "market_comparison": market_comparison,
                "stock_analysis": stock_analysis,
                "learning_points": learning_points,
                "simulation_metadata": {
                    "simulated_at": datetime.now().isoformat(),
                    "num_stocks": len(simulation_stocks),
                    "simulation_days": (sim_end_date - sim_start_date).days
                }
            }
            
            print(f"✅ 시뮬레이션 완료: 수익률 {portfolio_performance['total_return_pct']:.2f}%")
            return result
            
        except Exception as e:
            print(f"❌ 시뮬레이션 실패: {e}")
            traceback.print_exc()
            raise
    
    def _prepare_simulation_stocks(
        self, 
        selected_stocks: List[Dict[str, Any]], 
        total_amount: int
    ) -> List[SimulationStock]:
        """시뮬레이션용 종목 정보 준비"""
        simulation_stocks = []
        
        for stock in selected_stocks:
            allocation_pct = stock.get('allocation', 0)
            if allocation_pct <= 0:
                continue
                
            allocation_amount = int(total_amount * allocation_pct / 100)
            
            sim_stock = SimulationStock(
                code=stock['code'],
                name=stock['name'], 
                industry=stock.get('industry', 'Unknown'),
                allocation=allocation_pct,
                amount=allocation_amount
            )
            simulation_stocks.append(sim_stock)
        
        return simulation_stocks
    
    async def _fetch_stock_data(
        self,
        stocks: List[SimulationStock],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> Dict[str, pd.DataFrame]:
        """주가 데이터 수집"""
        stock_data = {}
        
        # 시작일을 조금 앞당겨서 데이터 수집 (휴일 대비)
        fetch_start = start_date - pd.DateOffset(days=10)
        fetch_end = end_date + pd.DateOffset(days=5)
        
        for stock in stocks:
            try:
                yahoo_code = self.stock_code_mapping.get(stock.code)
                if not yahoo_code:
                    print(f"⚠️ 종목 코드 매핑 없음: {stock.code} ({stock.name})")
                    continue
                
                print(f"📈 주가 데이터 수집: {stock.name} ({yahoo_code})")
                
                # yfinance로 데이터 수집
                ticker = yf.Ticker(yahoo_code)
                hist = ticker.history(
                    start=fetch_start.strftime('%Y-%m-%d'),
                    end=fetch_end.strftime('%Y-%m-%d'),
                    auto_adjust=True
                )
                
                if hist.empty:
                    print(f"⚠️ 데이터 없음: {stock.name}")
                    continue
                
                # 데이터 정리
                hist.index = pd.to_datetime(hist.index)
                hist = hist.loc[start_date:end_date]
                
                if not hist.empty:
                    stock_data[stock.code] = hist
                    print(f"✅ 수집 완료: {stock.name} ({len(hist)}일)")
                else:
                    print(f"⚠️ 기간 내 데이터 없음: {stock.name}")
                
                # API 호출 제한 방지
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"❌ {stock.name} 데이터 수집 실패: {e}")
                continue
        
        return stock_data
    
    async def _fetch_market_data(
        self,
        start_date: pd.Timestamp, 
        end_date: pd.Timestamp
    ) -> Dict[str, pd.DataFrame]:
        """시장 지수 데이터 수집"""
        market_data = {}
        
        fetch_start = start_date - pd.DateOffset(days=10)
        fetch_end = end_date + pd.DateOffset(days=5)
        
        for index_name, yahoo_code in self.market_indices.items():
            try:
                print(f"📊 시장 데이터 수집: {index_name}")
                
                ticker = yf.Ticker(yahoo_code)
                hist = ticker.history(
                    start=fetch_start.strftime('%Y-%m-%d'),
                    end=fetch_end.strftime('%Y-%m-%d'),
                    auto_adjust=True
                )
                
                if not hist.empty:
                    hist.index = pd.to_datetime(hist.index)
                    hist = hist.loc[start_date:end_date]
                    if not hist.empty:
                        market_data[index_name] = hist
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"❌ {index_name} 데이터 수집 실패: {e}")
                continue
        
        return market_data
    
    def _calculate_portfolio_performance(
        self,
        stocks: List[SimulationStock],
        stock_data: Dict[str, pd.DataFrame],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> Dict[str, Any]:
        """포트폴리오 성과 계산"""
        
        # 일별 포트폴리오 가치 계산
        portfolio_values = []
        dates = []
        
        # 사용 가능한 모든 날짜 수집
        all_dates = set()
        for code, data in stock_data.items():
            all_dates.update(data.index)
        
        all_dates = sorted(list(all_dates))
        
        initial_portfolio_value = sum(stock.amount for stock in stocks)
        
        for date in all_dates:
            if date < start_date or date > end_date:
                continue
                
            daily_value = 0
            valid_stocks = 0
            
            for stock in stocks:
                if stock.code not in stock_data:
                    continue
                    
                data = stock_data[stock.code]
                if date not in data.index:
                    continue
                
                # 해당 날짜의 종가 기준 가치 계산
                start_price = data.iloc[0]['Close'] if len(data) > 0 else None
                current_price = data.loc[date]['Close']
                
                if start_price and start_price > 0:
                    stock_return = (current_price / start_price) - 1
                    stock_value = stock.amount * (1 + stock_return)
                    daily_value += stock_value
                    valid_stocks += 1
            
            if valid_stocks > 0:
                portfolio_values.append(daily_value)
                dates.append(date)
        
        if not portfolio_values:
            raise ValueError("포트폴리오 가치 계산 실패: 유효한 데이터 없음")
        
        # 성과 지표 계산
        final_value = portfolio_values[-1]
        total_return = final_value - initial_portfolio_value
        total_return_pct = (total_return / initial_portfolio_value) * 100
        
        # 일일 수익률
        daily_returns = []
        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] / portfolio_values[i-1]) - 1
            daily_returns.append(daily_return)
        
        # 변동성 (연환산)
        volatility = np.std(daily_returns) * np.sqrt(252) * 100 if daily_returns else 0
        
        # 샤프 비율 (무위험 수익률 0% 가정)
        avg_daily_return = np.mean(daily_returns) if daily_returns else 0
        sharpe_ratio = (avg_daily_return * 252) / (volatility / 100) if volatility > 0 else 0
        
        return {
            "initial_value": initial_portfolio_value,
            "final_value": int(final_value),
            "total_return": int(total_return),
            "total_return_pct": round(total_return_pct, 2),
            "daily_values": portfolio_values,
            "dates": [d.strftime('%Y-%m-%d') for d in dates],
            "daily_returns": daily_returns,
            "volatility": round(volatility, 2),
            "sharpe_ratio": round(sharpe_ratio, 2)
        }
    
    def _analyze_individual_stocks(
        self,
        stocks: List[SimulationStock],
        stock_data: Dict[str, pd.DataFrame],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> List[Dict[str, Any]]:
        """종목별 성과 분석"""
        stock_analysis = []
        
        for stock in stocks:
            if stock.code not in stock_data:
                continue
                
            data = stock_data[stock.code]
            if data.empty:
                continue
            
            start_price = data.iloc[0]['Close']
            end_price = data.iloc[-1]['Close']
            
            stock_return = (end_price / start_price) - 1
            stock_return_pct = stock_return * 100
            profit_amount = int(stock.amount * stock_return)
            
            # 최고/최저 수익률
            prices = data['Close']
            max_price = prices.max()
            min_price = prices.min()
            max_return = (max_price / start_price) - 1
            min_return = (min_price / start_price) - 1
            
            analysis = {
                "code": stock.code,
                "name": stock.name,
                "industry": stock.industry,
                "investment_amount": stock.amount,
                "allocation_pct": stock.allocation,
                "start_price": round(start_price, 0),
                "end_price": round(end_price, 0),
                "return_pct": round(stock_return_pct, 2),
                "profit_amount": profit_amount,
                "max_return_pct": round(max_return * 100, 2),
                "min_return_pct": round(min_return * 100, 2),
                "volatility": round(prices.pct_change().std() * np.sqrt(252) * 100, 2)
            }
            
            stock_analysis.append(analysis)
        
        # 수익률 순으로 정렬
        stock_analysis.sort(key=lambda x: x['return_pct'], reverse=True)
        
        return stock_analysis
    
    def _compare_with_market(
        self,
        portfolio_performance: Dict[str, Any],
        market_data: Dict[str, pd.DataFrame],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> Dict[str, Any]:
        """시장 대비 성과 분석"""
        comparison = {}
        
        portfolio_return = portfolio_performance['total_return_pct']
        
        for index_name, data in market_data.items():
            if data.empty:
                continue
                
            start_value = data.iloc[0]['Close']
            end_value = data.iloc[-1]['Close']
            market_return = ((end_value / start_value) - 1) * 100
            
            alpha = portfolio_return - market_return
            
            comparison[index_name] = {
                "market_return_pct": round(market_return, 2),
                "alpha": round(alpha, 2),
                "outperformed": alpha > 0
            }
        
        return comparison
    
    def _generate_learning_points(
        self,
        scenario: SimulationScenario,
        portfolio_performance: Dict[str, Any],
        stock_analysis: List[Dict[str, Any]],
        market_comparison: Dict[str, Any]
    ) -> List[str]:
        """학습 포인트 생성"""
        learning_points = []
        
        total_return = portfolio_performance['total_return_pct']
        
        # 1. 전체 성과 평가
        if total_return > 10:
            learning_points.append(f"✅ 우수한 성과: 총 수익률 {total_return:.1f}%로 높은 수익을 달성했습니다.")
        elif total_return > 5:
            learning_points.append(f"✅ 양호한 성과: 총 수익률 {total_return:.1f}%로 안정적인 수익을 얻었습니다.")
        elif total_return > 0:
            learning_points.append(f"📊 보통 성과: 총 수익률 {total_return:.1f}%로 소폭 상승했습니다.")
        else:
            learning_points.append(f"⚠️ 손실 발생: 총 {abs(total_return):.1f}% 손실이 발생했습니다.")
        
        # 2. 시나리오별 특성 분석
        if scenario.id == "PN_005":  # 이란 사건
            learning_points.append("🛡️ 지정학적 리스크 시 방산주와 정유주가 수혜를 받는 패턴을 확인할 수 있습니다.")
        elif scenario.id == "PN_003":  # 탄소중립
            learning_points.append("🌱 정책 테마 투자: 정부 정책 발표는 관련 산업에 장기적 영향을 미칩니다.")
        elif scenario.id == "PN_002":  # 금리 상승
            learning_points.append("📈 금리 상승기에는 고PER 성장주보다 실적주가 유리한 경향을 보입니다.")
        
        # 3. 종목별 성과 분석
        if stock_analysis:
            best_stock = stock_analysis[0]
            worst_stock = stock_analysis[-1]
            
            learning_points.append(
                f"🏆 최고 수익 종목: {best_stock['name']} (+{best_stock['return_pct']:.1f}%)"
            )
            
            if worst_stock['return_pct'] < 0:
                learning_points.append(
                    f"📉 최대 손실 종목: {worst_stock['name']} ({worst_stock['return_pct']:.1f}%)"
                )
        
        # 4. 시장 대비 성과
        kospi_comparison = market_comparison.get('KOSPI', {})
        if kospi_comparison:
            alpha = kospi_comparison['alpha']
            if alpha > 0:
                learning_points.append(f"📊 시장 대비 우수: KOSPI 대비 +{alpha:.1f}%p 초과 수익")
            else:
                learning_points.append(f"📊 시장 대비 부진: KOSPI 대비 {alpha:.1f}%p 낮은 수익")
        
        # 5. 리스크 관리 포인트
        volatility = portfolio_performance['volatility']
        if volatility > 30:
            learning_points.append("⚠️ 높은 변동성: 리스크 관리와 분산투자가 더 필요했습니다.")
        elif volatility < 15:
            learning_points.append("✅ 안정적 투자: 낮은 변동성으로 안정적인 투자를 실행했습니다.")
        
        # 6. 추가 인사이트
        if len(stock_analysis) < 3:
            learning_points.append("💡 분산투자 권장: 더 많은 종목에 분산투자하면 리스크를 줄일 수 있습니다.")
        
        return learning_points

    async def get_scenario_info(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """시나리오 정보 조회"""
        scenario = self.scenarios.get(scenario_id)
        if not scenario:
            return None
        
        return {
            "id": scenario.id,
            "name": scenario.name,
            "description": scenario.description,
            "start_date": scenario.start_date,
            "end_date": scenario.end_date,
            "related_industries": scenario.related_industries,
            "impact_description": scenario.impact_description,
            "expected_sectors": scenario.expected_sectors
        }
    
    def get_available_scenarios(self) -> List[Dict[str, Any]]:
        """사용 가능한 시나리오 목록 반환"""
        scenarios_list = []
        
        for scenario in self.scenarios.values():
            scenarios_list.append({
                "id": scenario.id,
                "name": scenario.name,
                "description": scenario.description[:100] + "..." if len(scenario.description) > 100 else scenario.description,
                "period": f"{scenario.start_date} ~ {scenario.end_date}",
                "related_industries": scenario.related_industries
            })
        
        return scenarios_list
    
    def get_recommended_stocks_for_scenario(self, scenario_id: str) -> Dict[str, List[Dict[str, str]]]:
        """시나리오별 추천 종목 반환"""
        scenario = self.scenarios.get(scenario_id)
        if not scenario:
            return {}
        
        # 시나리오별 추천 종목 매핑
        recommended_stocks = {
            "PN_001": {  # 바이든 부양책
                "건설": [
                    {"code": "000720", "name": "현대건설"},
                    {"code": "028050", "name": "삼성물산"},
                ],
                "화학": [
                    {"code": "051910", "name": "LG화학"},
                    {"code": "096770", "name": "SK이노베이션"},
                ],
                "금속": [
                    {"code": "005490", "name": "POSCO홀딩스"},
                    {"code": "004020", "name": "현대제철"},
                ]
            },
            "PN_002": {  # 금리 상승
                "금융": [
                    {"code": "105560", "name": "KB금융"},
                    {"code": "086790", "name": "하나금융지주"},
                ]
            },
            "PN_003": {  # 탄소중립
                "전기전자": [
                    {"code": "005930", "name": "삼성전자"},
                    {"code": "373220", "name": "LG에너지솔루션"},
                ],
                "화학": [
                    {"code": "051910", "name": "LG화학"},
                    {"code": "096770", "name": "SK이노베이션"},
                ]
            },
            "PN_004": {  # 코로나 재확산
                "의료·정밀기기": [
                    {"code": "145020", "name": "휴젤"},
                    {"code": "196170", "name": "알테오젠"},
                ],
                "IT 서비스": [
                    {"code": "035720", "name": "카카오"},
                    {"code": "035420", "name": "네이버"},
                ]
            },
            "PN_005": {  # 이란 사건
                "정유": [
                    {"code": "010950", "name": "S-OIL"},
                    {"code": "078930", "name": "GS"},
                    {"code": "096770", "name": "SK이노베이션"},
                ],
                "방위산업": [
                    {"code": "047810", "name": "한국항공우주"},
                    {"code": "079550", "name": "LIG넥스원"},
                    {"code": "012450", "name": "한화시스템"},
                ]
            },
            "PN_006": {  # 일본 수출 규제
                "반도체": [
                    {"code": "005930", "name": "삼성전자"},
                    {"code": "000660", "name": "SK하이닉스"},
                ],
                "화학": [
                    {"code": "051910", "name": "LG화학"},
                    {"code": "096770", "name": "SK이노베이션"},
                ]
            }
        }
        
        return recommended_stocks.get(scenario_id, {})
    
    async def validate_simulation_inputs(
        self,
        scenario_id: str,
        investment_amount: int,
        investment_period: int,
        selected_stocks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """시뮬레이션 입력값 검증"""
        errors = []
        warnings = []
        
        # 시나리오 검증
        if scenario_id not in self.scenarios:
            errors.append(f"유효하지 않은 시나리오: {scenario_id}")
        
        # 투자 금액 검증
        if investment_amount < 100000:
            errors.append("최소 투자 금액은 10만원입니다.")
        elif investment_amount > 100000000:
            errors.append("최대 투자 금액은 1억원입니다.")
        
        # 투자 기간 검증
        if investment_period < 1:
            errors.append("최소 투자 기간은 1개월입니다.")
        elif investment_period > 24:
            errors.append("최대 투자 기간은 24개월입니다.")
        
        # 종목 검증
        if not selected_stocks:
            errors.append("최소 1개 이상의 종목을 선택해야 합니다.")
        elif len(selected_stocks) > 10:
            warnings.append("10개 이상 종목 선택 시 관리가 어려울 수 있습니다.")
        
        # 투자 비중 검증
        total_allocation = sum(stock.get('allocation', 0) for stock in selected_stocks)
        if abs(total_allocation - 100) > 5:  # 5% 오차 허용
            warnings.append(f"총 투자 비중이 {total_allocation}%입니다. 100%에 맞춰주세요.")
        
        # 종목 코드 검증
        for stock in selected_stocks:
            code = stock.get('code', '')
            if code not in self.stock_code_mapping:
                warnings.append(f"종목 코드 매핑 없음: {code} ({stock.get('name', 'Unknown')})")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def save_simulation_result(self, result: Dict[str, Any], filename: Optional[str] = None) -> str:
        """시뮬레이션 결과 저장"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scenario_id = result.get("scenario_info", {}).get("id", "unknown")
            filename = f"simulation_{scenario_id}_{timestamp}.json"
        
        filepath = self.data_dir / filename
        
        # JSON serializable하도록 변환
        serializable_result = self._make_json_serializable(result)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
        
        print(f"💾 시뮬레이션 결과 저장: {filepath}")
        return str(filepath)
    
    def _make_json_serializable(self, obj):
        """JSON 직렬화 가능하도록 변환"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj

# ===== 유틸리티 함수들 =====

async def create_simple_simulation(
    scenario_id: str = "PN_005",
    stocks_config: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """간단한 시뮬레이션 생성 (테스트용)"""
    
    if stocks_config is None:
        stocks_config = [
            {"code": "010950", "name": "S-OIL", "industry": "정유", "allocation": 40},
            {"code": "047810", "name": "한국항공우주", "industry": "방위산업", "allocation": 35},
            {"code": "105560", "name": "KB금융", "industry": "금융", "allocation": 25}
        ]
    
    engine = SimulationEngine()
    
    result = await engine.run_simulation(
        scenario_id=scenario_id,
        investment_amount=1000000,  # 100만원
        investment_period=3,        # 3개월
        selected_stocks=stocks_config
    )
    
    return result

def get_simulation_summary(result: Dict[str, Any]) -> str:
    """시뮬레이션 결과 요약"""
    scenario = result["scenario_info"]
    performance = result["simulation_results"]
    
    summary = f"""
🎯 {scenario['name']} 시뮬레이션 결과

📊 투자 성과:
• 투자 금액: {performance['initial_amount']:,}원
• 최종 금액: {performance['final_amount']:,}원  
• 총 수익률: {performance['total_return_pct']:+.2f}%
• 수익 금액: {performance['total_return']:+,}원

📈 리스크 지표:
• 변동성: {performance['volatility']:.2f}%
• 샤프 비율: {performance['sharpe_ratio']:.2f}

🎓 학습 포인트:
"""
    
    for point in result["learning_points"]:
        summary += f"• {point}\n"
    
    return summary

# ===== 테스트 및 직접 실행용 =====

async def test_simulation_engine():
    """시뮬레이션 엔진 테스트"""
    print("🧪 모의투자 시뮬레이션 엔진 테스트")
    
    try:
        # 시뮬레이션 엔진 초기화
        engine = SimulationEngine()
        
        # 1. 사용 가능한 시나리오 확인
        print("\n1️⃣ 사용 가능한 시나리오:")
        scenarios = engine.get_available_scenarios()
        for scenario in scenarios[:3]:  # 처음 3개만 출력
            print(f"  • {scenario['id']}: {scenario['name']}")
        
        # 2. 특정 시나리오 정보 조회
        print("\n2️⃣ 시나리오 상세 정보:")
        scenario_info = await engine.get_scenario_info("PN_005")
        if scenario_info:
            print(f"  • 이름: {scenario_info['name']}")
            print(f"  • 기간: {scenario_info['start_date']} ~ {scenario_info['end_date']}")
            print(f"  • 관련 산업: {', '.join(scenario_info['related_industries'])}")
        
        # 3. 추천 종목 조회
        print("\n3️⃣ 추천 종목:")
        recommended = engine.get_recommended_stocks_for_scenario("PN_005")
        for industry, stocks in recommended.items():
            print(f"  • {industry}: {[s['name'] for s in stocks]}")
        
        # 4. 시뮬레이션 실행
        print("\n4️⃣ 시뮬레이션 실행:")
        test_stocks = [
            {"code": "010950", "name": "S-OIL", "industry": "정유", "allocation": 50},
            {"code": "047810", "name": "한국항공우주", "industry": "방위산업", "allocation": 50}
        ]
        
        result = await engine.run_simulation(
            scenario_id="PN_005",
            investment_amount=1000000,
            investment_period=3,
            selected_stocks=test_stocks
        )
        
        # 5. 결과 요약 출력
        print("\n5️⃣ 시뮬레이션 결과:")
        summary = get_simulation_summary(result)
        print(summary)
        
        # 6. 결과 저장
        print("\n6️⃣ 결과 저장:")
        saved_file = engine.save_simulation_result(result)
        print(f"저장 완료: {saved_file}")
        
        print("\n✅ 모든 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # 직접 실행 시 테스트
    asyncio.run(test_simulation_engine())