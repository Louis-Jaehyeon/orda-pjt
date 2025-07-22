from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, Any]
    version: str

class AnalysisRequest(BaseModel):
    news_content: str
    issue_id: Optional[int] = None
    source: str = "manual"

class AnalysisResponse(BaseModel):
    current_news: str
    similar_past_issues: List[Dict[str, Any]]
    related_industries: List[Dict[str, Any]]
    analysis_result: str
    confidence_score: float

class SimulationRequest(BaseModel):
    scenario_id: str
    investment_amount: int
    investment_period: int  # 단위: months
    selected_stocks: List[Dict[str, Any]]

class SimulationResponse(BaseModel):
    scenario_info: Dict[str, Any]
    simulation_results: Dict[str, Any]
    market_comparison: Dict[str, Any]
    stock_analysis: List[Dict[str, Any]]
    learning_points: List[str]
    simulation_metadata: Dict[str, Any]
