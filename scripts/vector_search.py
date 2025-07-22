#!/usr/bin/env python3
"""
벡터 검색 엔진 모듈 - Pinecone 통합 관리
모든 네임스페이스의 벡터 검색 및 관리를 담당
"""

import os
import asyncio
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import traceback
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

class VectorSearchEngine:
    """Pinecone 벡터 검색 및 관리 엔진"""
    
    def __init__(self, index_name: str = "lastproject"):
        """
        초기화
        
        Args:
            index_name: Pinecone 인덱스 이름
        """
        load_dotenv(override=True)
        
        self.index_name = index_name
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        if not self.openai_api_key or not self.pinecone_api_key:
            raise ValueError("OPENAI_API_KEY 또는 PINECONE_API_KEY가 설정되지 않았습니다.")
        
        # 임베딩 모델 초기화
        self.embedding = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.openai_api_key
        )
        
        # Pinecone 클라이언트 초기화
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # 인덱스 연결
        self._connect_index()
        
        # 네임스페이스별 벡터 스토어 초기화
        self._init_vector_stores()
    
    def _connect_index(self):
        """Pinecone 인덱스 연결"""
        try:
            # 인덱스 존재 확인
            indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in indexes:
                print(f"⚠️ 인덱스 '{self.index_name}'가 존재하지 않습니다.")
                print("💡 인덱스를 생성하시겠습니까? (자동 생성 시도)")
                self._create_index()
            
            self.index = self.pc.Index(self.index_name)
            print(f"✅ Pinecone 인덱스 연결 완료: {self.index_name}")
            
        except Exception as e:
            print(f"❌ Pinecone 인덱스 연결 실패: {e}")
            raise
    
    def _create_index(self):
        """인덱스 자동 생성"""
        try:
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # text-embedding-3-small 차원
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"✅ 인덱스 '{self.index_name}' 생성 완료")
            
            # 인덱스 초기화 대기
            import time
            time.sleep(30)
            
        except Exception as e:
            print(f"❌ 인덱스 생성 실패: {e}")
            raise
    
    def _init_vector_stores(self):
        """네임스페이스별 벡터 스토어 초기화"""
        self.namespaces = {
            "industry": "산업분류 정보",
            "past_issue": "과거 이슈 정보", 
            "current_issue": "현재 이슈 정보"
        }
        
        self.vector_stores = {}
        
        for namespace, description in self.namespaces.items():
            try:
                self.vector_stores[namespace] = PineconeVectorStore(
                    index_name=self.index_name,
                    embedding=self.embedding,
                    namespace=namespace
                )
                print(f"✅ {namespace} 벡터 스토어 초기화 완료 ({description})")
                
            except Exception as e:
                print(f"⚠️ {namespace} 벡터 스토어 초기화 실패: {e}")
    
    async def search_similar_past_issues(
        self, 
        query: str, 
        top_k: int = 3,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        과거 유사 이슈 검색
        
        Args:
            query: 검색어 (현재 뉴스 내용)
            top_k: 반환할 최대 결과 수
            similarity_threshold: 유사도 임계값
            
        Returns:
            유사 이슈 리스트
        """
        try:
            print(f"🔍 과거 이슈 검색: '{query[:50]}...'")
            
            if "past_issue" not in self.vector_stores:
                print("⚠️ past_issue 벡터 스토어가 초기화되지 않았습니다.")
                return []
            
            # 유사도 검색 실행
            docs = self.vector_stores["past_issue"].similarity_search_with_score(
                query, k=top_k
            )
            
            results = []
            for doc, score in docs:
                # 유사도 임계값 확인 (score가 낮을수록 유사함)
                similarity = 1 - score  # cosine similarity로 변환
                
                if similarity >= similarity_threshold:
                    result = {
                        "document": doc,
                        "similarity_score": similarity,
                        "parsed_content": self._parse_past_issue_document(doc),
                        "metadata": doc.metadata
                    }
                    results.append(result)
            
            print(f"✅ {len(results)}개 유사 이슈 발견 (임계값: {similarity_threshold})")
            return results
            
        except Exception as e:
            print(f"❌ 과거 이슈 검색 실패: {e}")
            traceback.print_exc()
            return []
    
    async def search_related_industries(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        관련 산업 검색
        
        Args:
            query: 검색어
            top_k: 반환할 최대 결과 수  
            similarity_threshold: 유사도 임계값
            
        Returns:
            관련 산업 리스트
        """
        try:
            print(f"🏭 관련 산업 검색: '{query[:50]}...'")
            
            if "industry" not in self.vector_stores:
                print("⚠️ industry 벡터 스토어가 초기화되지 않았습니다.")
                return []
            
            # 유사도 검색 실행
            docs = self.vector_stores["industry"].similarity_search_with_score(
                query, k=top_k
            )
            
            results = []
            seen_industries = set()  # 중복 산업 제거용
            
            for doc, score in docs:
                similarity = 1 - score
                
                if similarity >= similarity_threshold:
                    parsed = self._parse_industry_document(doc)
                    industry_name = parsed.get("industry_name", "미확인")
                    
                    # 중복 산업 제거
                    if industry_name not in seen_industries:
                        seen_industries.add(industry_name)
                        
                        result = {
                            "document": doc,
                            "similarity_score": similarity,
                            "parsed_content": parsed,
                            "metadata": doc.metadata
                        }
                        results.append(result)
            
            print(f"✅ {len(results)}개 관련 산업 발견")
            return results
            
        except Exception as e:
            print(f"❌ 관련 산업 검색 실패: {e}")
            traceback.print_exc()
            return []
    
    async def search_current_issues(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        현재 이슈 검색 (크롤링된 최신 데이터에서)
        
        Args:
            query: 검색어
            top_k: 반환할 최대 결과 수
            
        Returns:
            현재 이슈 리스트
        """
        try:
            print(f"📰 현재 이슈 검색: '{query[:50]}...'")
            
            if "current_issue" not in self.vector_stores:
                print("⚠️ current_issue 벡터 스토어가 없습니다. 빈 결과 반환.")
                return []
            
            docs = self.vector_stores["current_issue"].similarity_search_with_score(
                query, k=top_k
            )
            
            results = []
            for doc, score in docs:
                result = {
                    "document": doc,
                    "similarity_score": 1 - score,
                    "parsed_content": self._parse_current_issue_document(doc),
                    "metadata": doc.metadata
                }
                results.append(result)
            
            print(f"✅ {len(results)}개 현재 이슈 발견")
            return results
            
        except Exception as e:
            print(f"❌ 현재 이슈 검색 실패: {e}")
            return []
    
    def _parse_past_issue_document(self, doc: Document) -> Dict[str, str]:
        """과거 이슈 문서 파싱"""
        content_lines = doc.page_content.split('\n')
        
        parsed = {
            "issue_id": "미확인",
            "issue_name": "미확인",
            "contents": "미확인",
            "related_industries": "미확인",
            "industry_reason": "미확인",
            "start_date": "미확인",
            "end_date": "미확인",
            "source": "미확인"
        }
        
        for line in content_lines:
            line = line.strip()
            if line.startswith('ID:') or line.startswith('﻿ID:'):
                parsed["issue_id"] = line.replace('ID:', '').replace('﻿ID:', '').strip()
            elif line.startswith('Issue_name:'):
                parsed["issue_name"] = line.replace('Issue_name:', '').strip()
            elif line.startswith('Contents:'):
                parsed["contents"] = line.replace('Contents:', '').strip()
            elif line.startswith('관련 산업:'):
                parsed["related_industries"] = line.replace('관련 산업:', '').strip()
            elif line.startswith('산업 이유:'):
                parsed["industry_reason"] = line.replace('산업 이유:', '').strip()
            elif line.startswith('Start_date:'):
                parsed["start_date"] = line.replace('Start_date:', '').strip()
            elif line.startswith('Fin_date:'):
                parsed["end_date"] = line.replace('Fin_date:', '').strip()
            elif line.startswith('근거자료:'):
                parsed["source"] = line.replace('근거자료:', '').strip()
        
        return parsed
    
    def _parse_industry_document(self, doc: Document) -> Dict[str, str]:
        """산업 분류 문서 파싱"""
        content_lines = doc.page_content.split('\n')
        
        parsed = {
            "industry_name": "미확인",
            "description": "",
            "full_content": doc.page_content
        }
        
        for line in content_lines:
            line = line.strip()
            if line.startswith('KRX 업종명:') or line.startswith('﻿KRX 업종명:'):
                parsed["industry_name"] = line.replace('KRX 업종명:', '').replace('﻿KRX 업종명:', '').strip()
            elif line.startswith('상세내용:'):
                parsed["description"] = line.replace('상세내용:', '').strip()
                break  # 상세내용은 첫 번째 줄만 사용
        
        return parsed
    
    def _parse_current_issue_document(self, doc: Document) -> Dict[str, str]:
        """현재 이슈 문서 파싱"""
        content_lines = doc.page_content.split('\n')
        
        parsed = {
            "title": "미확인",
            "content": "미확인",
            "issue_number": "미확인"
        }
        
        # 간단한 파싱 (BigKinds 크롤링 결과 형태)
        full_content = doc.page_content
        if '\n' in full_content:
            lines = full_content.split('\n')
            if len(lines) >= 2:
                parsed["title"] = lines[0]
                parsed["content"] = '\n'.join(lines[1:])
        
        return parsed
    
    async def update_current_issues(self, issues_data: List[Dict]) -> bool:
        """
        현재 이슈 벡터 업데이트 (BigKinds 크롤링 결과)
        
        Args:
            issues_data: 크롤링된 이슈 데이터 리스트
            
        Returns:
            성공 여부
        """
        try:
            print(f"🔄 현재 이슈 벡터 업데이트 시작: {len(issues_data)}개")
            
            # 기존 current_issue 네임스페이스 삭제
            try:
                self.index.delete(delete_all=True, namespace="current_issue")
                print("✅ 기존 current_issue 벡터 삭제 완료")
            except Exception as e:
                print(f"⚠️ 기존 벡터 삭제 중 오류 (무시): {e}")
            
            # Document 객체로 변환
            documents = []
            for issue in issues_data:
                content = f"{issue.get('제목', '')}\n{issue.get('내용', '')}"
                metadata = {
                    "issue_id": issue.get('이슈번호', 0),
                    "title": issue.get('제목', ''),
                    "source": "bigkinds_crawling",
                    "crawled_at": datetime.now().isoformat()
                }
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=450,
                chunk_overlap=45,
                length_function=len
            )
            
            chunks = text_splitter.split_documents(documents)
            print(f"📄 {len(chunks)}개 청크 생성됨")
            
            # Pinecone에 업로드
            if chunks:
                self.vector_stores["current_issue"] = PineconeVectorStore.from_documents(
                    chunks,
                    embedding=self.embedding,
                    index_name=self.index_name,
                    namespace="current_issue"
                )
                print("✅ 현재 이슈 벡터 업데이트 완료")
                return True
            else:
                print("⚠️ 업로드할 데이터가 없습니다.")
                return False
            
        except Exception as e:
            print(f"❌ 현재 이슈 벡터 업데이트 실패: {e}")
            traceback.print_exc()
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """인덱스 통계 정보 반환"""
        try:
            stats = self.index.describe_index_stats()
            
            # 네임스페이스별 정보 정리
            namespace_info = {}
            namespaces = stats.get('namespaces', {})
            
            for ns_name, description in self.namespaces.items():
                ns_stats = namespaces.get(ns_name, {})
                namespace_info[ns_name] = {
                    "description": description,
                    "vector_count": ns_stats.get('vector_count', 0),
                    "status": "active" if ns_stats.get('vector_count', 0) > 0 else "empty"
                }
            
            return {
                "index_name": self.index_name,
                "total_vectors": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 1536),
                "namespaces": namespace_info,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ 인덱스 통계 조회 실패: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """벡터 검색 엔진 상태 체크"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {}
        }
        
        try:
            # 1. Pinecone 연결 상태
            try:
                stats = self.index.describe_index_stats()
                health_status["components"]["pinecone"] = {
                    "status": "healthy",
                    "total_vectors": stats.get('total_vector_count', 0)
                }
            except Exception as e:
                health_status["components"]["pinecone"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["overall_status"] = "degraded"
            
            # 2. 네임스페이스별 상태
            for namespace in self.namespaces.keys():
                try:
                    # 간단한 검색 테스트
                    test_results = await asyncio.to_thread(
                        self.vector_stores[namespace].similarity_search,
                        "test query", k=1
                    )
                    
                    health_status["components"][f"namespace_{namespace}"] = {
                        "status": "healthy",
                        "searchable": len(test_results) >= 0
                    }
                except Exception as e:
                    health_status["components"][f"namespace_{namespace}"] = {
                        "status": "unhealthy", 
                        "error": str(e)
                    }
                    health_status["overall_status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "unhealthy",
                "error": str(e)
            }

# ===== 유틸리티 함수들 =====

async def batch_search(
    engine: VectorSearchEngine,
    queries: List[str],
    search_type: str = "past_issue"
) -> List[List[Dict]]:
    """
    배치 검색 (여러 쿼리 동시 처리)
    
    Args:
        engine: VectorSearchEngine 인스턴스
        queries: 검색어 리스트
        search_type: 검색 타입 ("past_issue", "industry", "current_issue")
        
    Returns:
        검색 결과 리스트의 리스트
    """
    search_functions = {
        "past_issue": engine.search_similar_past_issues,
        "industry": engine.search_related_industries,
        "current_issue": engine.search_current_issues
    }
    
    if search_type not in search_functions:
        raise ValueError(f"지원하지 않는 검색 타입: {search_type}")
    
    search_func = search_functions[search_type]
    
    # 비동기 배치 실행
    tasks = [search_func(query) for query in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 예외 처리
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"⚠️ 쿼리 {i+1} 검색 실패: {result}")
            processed_results.append([])
        else:
            processed_results.append(result)
    
    return processed_results

# ===== 테스트 및 직접 실행용 =====

async def test_vector_search():
    """벡터 검색 엔진 테스트"""
    print("🧪 벡터 검색 엔진 테스트")
    
    try:
        # 엔진 초기화
        engine = VectorSearchEngine()
        
        # 1. 상태 체크
        print("\n1️⃣ 상태 체크")
        health = await engine.health_check()
        print(f"전체 상태: {health['overall_status']}")
        
        # 2. 인덱스 통계
        print("\n2️⃣ 인덱스 통계")
        stats = engine.get_index_stats()
        print(f"총 벡터 수: {stats.get('total_vectors', 0)}")
        for ns, info in stats.get('namespaces', {}).items():
            print(f"  • {ns}: {info['vector_count']}개 ({info['status']})")
        
        # 3. 검색 테스트
        test_query = "SK텔레콤 고객 이탈 보안 사고"
        
        print(f"\n3️⃣ 검색 테스트: '{test_query}'")
        
        # 과거 이슈 검색
        past_issues = await engine.search_similar_past_issues(test_query, top_k=2)
        print(f"과거 이슈: {len(past_issues)}개 발견")
        for issue in past_issues:
            parsed = issue["parsed_content"]
            print(f"  • {parsed['issue_name']} (유사도: {issue['similarity_score']:.2f})")
        
        # 관련 산업 검색
        industries = await engine.search_related_industries(test_query, top_k=2)
        print(f"관련 산업: {len(industries)}개 발견")
        for industry in industries:
            parsed = industry["parsed_content"]
            print(f"  • {parsed['industry_name']} (유사도: {industry['similarity_score']:.2f})")
        
        print("\n✅ 모든 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # 직접 실행 시 테스트
    asyncio.run(test_vector_search())