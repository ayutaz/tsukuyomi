"""API ドキュメント自動生成

OpenAPI/Swagger仕様のドキュメント生成
"""

from typing import Dict, List, Optional
from pathlib import Path
import json

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


# リクエスト/レスポンスモデル
class TTSRequest(BaseModel):
    """音声合成リクエスト"""
    text: str = Field(..., description="合成するテキスト", example="こんにちは、音声合成のテストです。")
    speaker_id: int = Field(0, description="話者ID", ge=0, lt=100)
    language: Optional[str] = Field("ja", description="言語コード（ja, en, zh）", pattern="^(ja|en|zh)$")
    speed: float = Field(1.0, description="話速（0.5-2.0）", ge=0.5, le=2.0)
    pitch_shift: float = Field(0.0, description="ピッチシフト（-12.0-12.0）", ge=-12.0, le=12.0)
    energy: float = Field(1.0, description="エネルギー/音量（0.5-2.0）", ge=0.5, le=2.0)
    
    class Config:
        schema_extra = {
            "example": {
                "text": "今日はいい天気ですね。",
                "speaker_id": 0,
                "language": "ja",
                "speed": 1.0,
                "pitch_shift": 0.0,
                "energy": 1.0,
            }
        }


class TTSBatchRequest(BaseModel):
    """バッチ音声合成リクエスト"""
    items: List[TTSRequest] = Field(..., description="合成リクエストのリスト", max_items=32)
    
    class Config:
        schema_extra = {
            "example": {
                "items": [
                    {
                        "text": "最初のテキストです。",
                        "speaker_id": 0,
                    },
                    {
                        "text": "2番目のテキストです。",
                        "speaker_id": 1,
                    },
                ]
            }
        }


class TTSResponse(BaseModel):
    """音声合成レスポンス"""
    audio_url: str = Field(..., description="生成された音声ファイルのURL")
    duration: float = Field(..., description="音声の長さ（秒）")
    sample_rate: int = Field(48000, description="サンプリングレート")
    request_id: str = Field(..., description="リクエストID")
    processing_time: float = Field(..., description="処理時間（秒）")


class TTSBatchResponse(BaseModel):
    """バッチ音声合成レスポンス"""
    results: List[TTSResponse] = Field(..., description="合成結果のリスト")
    total_processing_time: float = Field(..., description="総処理時間（秒）")


class ErrorResponse(BaseModel):
    """エラーレスポンス"""
    error: str = Field(..., description="エラーメッセージ")
    detail: Optional[str] = Field(None, description="詳細情報")
    request_id: Optional[str] = Field(None, description="リクエストID")


class SpeakerInfo(BaseModel):
    """話者情報"""
    id: int = Field(..., description="話者ID")
    name: str = Field(..., description="話者名")
    language: str = Field(..., description="対応言語")
    gender: str = Field(..., description="性別", pattern="^(male|female|neutral)$")
    description: Optional[str] = Field(None, description="話者の説明")
    sample_url: Optional[str] = Field(None, description="サンプル音声URL")


class ModelInfo(BaseModel):
    """モデル情報"""
    name: str = Field(..., description="モデル名")
    version: str = Field(..., description="バージョン")
    languages: List[str] = Field(..., description="対応言語リスト")
    max_text_length: int = Field(..., description="最大テキスト長")
    sample_rate: int = Field(..., description="サンプリングレート")


class HealthStatus(BaseModel):
    """ヘルスチェックステータス"""
    status: str = Field(..., description="ステータス（healthy/unhealthy）")
    model_loaded: bool = Field(..., description="モデルロード状態")
    gpu_available: bool = Field(..., description="GPU利用可能状態")
    version: str = Field(..., description="APIバージョン")
    uptime: float = Field(..., description="稼働時間（秒）")


def setup_api_documentation(app: FastAPI):
    """APIドキュメントの設定"""
    
    # カスタムOpenAPIスキーマ
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
            
        openapi_schema = get_openapi(
            title="Tsukuyomi TTS API",
            version="1.0.0",
            description="""
# Tsukuyomi TTS API

最高峰の日本語音声合成システムのAPIドキュメント

## 主な機能

- **高品質音声合成**: 自然で表現豊かな音声を生成
- **多言語対応**: 日本語、英語、中国語に対応
- **話者選択**: 100人以上の話者から選択可能
- **リアルタイム処理**: 低レイテンシでの音声生成
- **バッチ処理**: 複数テキストの同時処理

## 認証

APIキーまたはJWTトークンによる認証が必要です。

```
Authorization: Bearer YOUR_API_KEY
```

## レート制限

- 通常: 100リクエスト/分
- プレミアム: 1000リクエスト/分

## エラーハンドリング

標準的なHTTPステータスコードを使用:
- 200: 成功
- 400: 不正なリクエスト
- 401: 認証エラー
- 429: レート制限超過
- 500: サーバーエラー
            """,
            routes=app.routes,
            servers=[
                {"url": "https://api.tsukuyomi-tts.com", "description": "本番環境"},
                {"url": "https://staging-api.tsukuyomi-tts.com", "description": "ステージング環境"},
                {"url": "http://localhost:8000", "description": "開発環境"},
            ],
        )
        
        # セキュリティスキーマの追加
        openapi_schema["components"]["securitySchemes"] = {
            "apiKey": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "APIキー認証",
            },
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT認証",
            },
        }
        
        # グローバルセキュリティ
        openapi_schema["security"] = [
            {"apiKey": []},
            {"bearerAuth": []},
        ]
        
        # タグの追加
        openapi_schema["tags"] = [
            {
                "name": "synthesis",
                "description": "音声合成エンドポイント",
            },
            {
                "name": "speakers",
                "description": "話者管理",
            },
            {
                "name": "models",
                "description": "モデル情報",
            },
            {
                "name": "health",
                "description": "ヘルスチェック",
            },
        ]
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
        
    app.openapi = custom_openapi
    
    # Swagger UIのカスタマイズ
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return HTMLResponse(content=f"""
<!DOCTYPE html>
<html>
<head>
    <title>Tsukuyomi TTS API - Swagger UI</title>
    <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css">
    <style>
        .swagger-ui .topbar {{ display: none }}
        .swagger-ui .info {{ margin-bottom: 20px }}
        .swagger-ui .info .title {{ color: #6366f1 }}
        .swagger-ui .btn.authorize {{ background-color: #6366f1; border-color: #6366f1 }}
        .swagger-ui .btn.authorize:hover {{ background-color: #4f46e5; border-color: #4f46e5 }}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-standalone-preset.js"></script>
    <script>
    window.onload = function() {{
        window.ui = SwaggerUIBundle({{
            url: "/openapi.json",
            dom_id: '#swagger-ui',
            deepLinking: true,
            presets: [
                SwaggerUIBundle.presets.apis,
                SwaggerUIStandalonePreset
            ],
            plugins: [
                SwaggerUIBundle.plugins.DownloadUrl
            ],
            layout: "StandaloneLayout",
            tryItOutEnabled: true,
            supportedSubmitMethods: ['get', 'post', 'put', 'delete', 'patch'],
            onComplete: function() {{
                console.log("Swagger UI loaded");
            }}
        }})
    }}
    </script>
</body>
</html>
        """, status_code=200)
        
    # ReDocのカスタマイズ
    @app.get("/redoc", include_in_schema=False)
    async def redoc_html():
        return HTMLResponse(content=f"""
<!DOCTYPE html>
<html>
<head>
    <title>Tsukuyomi TTS API - ReDoc</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ margin: 0; padding: 0; }}
        #redoc-container .menu-content {{ background-color: #1e293b; }}
        #redoc-container a {{ color: #6366f1; }}
    </style>
</head>
<body>
    <div id="redoc-container"></div>
    <script src="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js"></script>
    <script>
        Redoc.init('/openapi.json', {{
            scrollYOffset: 50,
            theme: {{
                colors: {{
                    primary: {{
                        main: '#6366f1'
                    }}
                }},
                typography: {{
                    fontSize: '16px',
                    headings: {{
                        fontFamily: 'system-ui, -apple-system, sans-serif'
                    }}
                }}
            }}
        }}, document.getElementById('redoc-container'))
    </script>
</body>
</html>
        """, status_code=200)


def generate_client_sdk(openapi_spec: Dict, language: str, output_dir: Path):
    """クライアントSDKの生成"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if language == "python":
        # Python SDKの生成
        sdk_content = '''"""
Tsukuyomi TTS Python Client SDK

自動生成されたクライアントライブラリ
"""

from typing import List, Optional, Dict, Any
import requests
from dataclasses import dataclass


@dataclass
class TTSRequest:
    text: str
    speaker_id: int = 0
    language: str = "ja"
    speed: float = 1.0
    pitch_shift: float = 0.0
    energy: float = 1.0


class TsukuyomiClient:
    """Tsukuyomi TTS APIクライアント"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.tsukuyomi-tts.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        
    def synthesize(self, request: TTSRequest) -> Dict[str, Any]:
        """音声合成"""
        response = self.session.post(
            f"{self.base_url}/v1/synthesis",
            json=request.__dict__,
        )
        response.raise_for_status()
        return response.json()
        
    def synthesize_batch(self, requests: List[TTSRequest]) -> Dict[str, Any]:
        """バッチ音声合成"""
        response = self.session.post(
            f"{self.base_url}/v1/synthesis/batch",
            json={"items": [r.__dict__ for r in requests]},
        )
        response.raise_for_status()
        return response.json()
        
    def get_speakers(self) -> List[Dict[str, Any]]:
        """話者リスト取得"""
        response = self.session.get(f"{self.base_url}/v1/speakers")
        response.raise_for_status()
        return response.json()
        
    def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


# 使用例
if __name__ == "__main__":
    client = TsukuyomiClient(api_key="YOUR_API_KEY")
    
    # 単一の音声合成
    request = TTSRequest(text="こんにちは、世界！")
    result = client.synthesize(request)
    print(f"Audio URL: {result['audio_url']}")
    
    # バッチ合成
    requests = [
        TTSRequest(text="最初のテキスト", speaker_id=0),
        TTSRequest(text="2番目のテキスト", speaker_id=1),
    ]
    results = client.synthesize_batch(requests)
    print(f"Generated {len(results['results'])} audio files")
'''
        
        with open(output_dir / "tsukuyomi_client.py", "w") as f:
            f.write(sdk_content)
            
    elif language == "typescript":
        # TypeScript SDKの生成
        sdk_content = '''/**
 * Tsukuyomi TTS TypeScript Client SDK
 * 
 * 自動生成されたクライアントライブラリ
 */

export interface TTSRequest {
  text: string;
  speakerId?: number;
  language?: string;
  speed?: number;
  pitchShift?: number;
  energy?: number;
}

export interface TTSResponse {
  audioUrl: string;
  duration: number;
  sampleRate: number;
  requestId: string;
  processingTime: number;
}

export interface SpeakerInfo {
  id: number;
  name: string;
  language: string;
  gender: string;
  description?: string;
  sampleUrl?: string;
}

export class TsukuyomiClient {
  private apiKey: string;
  private baseUrl: string;

  constructor(apiKey: string, baseUrl: string = 'https://api.tsukuyomi-tts.com') {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
  }

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  async synthesize(request: TTSRequest): Promise<TTSResponse> {
    return this.request<TTSResponse>('/v1/synthesis', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async synthesizeBatch(requests: TTSRequest[]): Promise<{ results: TTSResponse[] }> {
    return this.request('/v1/synthesis/batch', {
      method: 'POST',
      body: JSON.stringify({ items: requests }),
    });
  }

  async getSpeakers(): Promise<SpeakerInfo[]> {
    return this.request<SpeakerInfo[]>('/v1/speakers');
  }

  async healthCheck(): Promise<{ status: string }> {
    return this.request('/health');
  }
}

// 使用例
async function example() {
  const client = new TsukuyomiClient('YOUR_API_KEY');
  
  // 単一の音声合成
  const result = await client.synthesize({
    text: 'こんにちは、世界！',
    speakerId: 0,
  });
  console.log(`Audio URL: ${result.audioUrl}`);
  
  // バッチ合成
  const batchResults = await client.synthesizeBatch([
    { text: '最初のテキスト', speakerId: 0 },
    { text: '2番目のテキスト', speakerId: 1 },
  ]);
  console.log(`Generated ${batchResults.results.length} audio files`);
}
'''
        
        with open(output_dir / "tsukuyomi-client.ts", "w") as f:
            f.write(sdk_content)
            
    print(f"Generated {language} SDK in {output_dir}")


def generate_postman_collection(openapi_spec: Dict, output_path: Path):
    """Postmanコレクションの生成"""
    collection = {
        "info": {
            "name": "Tsukuyomi TTS API",
            "description": openapi_spec.get("info", {}).get("description", ""),
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "auth": {
            "type": "bearer",
            "bearer": [{"key": "token", "value": "{{api_key}}", "type": "string"}],
        },
        "variable": [
            {"key": "base_url", "value": "https://api.tsukuyomi-tts.com"},
            {"key": "api_key", "value": "YOUR_API_KEY"},
        ],
        "item": [],
    }
    
    # エンドポイントの変換
    for path, methods in openapi_spec.get("paths", {}).items():
        for method, operation in methods.items():
            if method in ["get", "post", "put", "delete", "patch"]:
                item = {
                    "name": operation.get("summary", path),
                    "request": {
                        "method": method.upper(),
                        "header": [],
                        "url": {
                            "raw": "{{base_url}}" + path,
                            "host": ["{{base_url}}"],
                            "path": path.strip("/").split("/"),
                        },
                    },
                }
                
                # リクエストボディの例
                if "requestBody" in operation:
                    content = operation["requestBody"].get("content", {})
                    if "application/json" in content:
                        schema = content["application/json"].get("schema", {})
                        if "example" in schema:
                            item["request"]["body"] = {
                                "mode": "raw",
                                "raw": json.dumps(schema["example"], indent=2),
                                "options": {"raw": {"language": "json"}},
                            }
                            
                collection["item"].append(item)
                
    # 保存
    with open(output_path, "w") as f:
        json.dump(collection, f, indent=2)
        
    print(f"Generated Postman collection: {output_path}")