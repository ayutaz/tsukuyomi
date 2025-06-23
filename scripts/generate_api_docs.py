#!/usr/bin/env python3
"""APIドキュメント生成スクリプト"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.server.api import create_app
from src.server.api_docs import generate_client_sdk, generate_postman_collection


def main():
    parser = argparse.ArgumentParser(description="APIドキュメント生成")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/api",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["openapi", "postman", "sdk-python", "sdk-typescript", "all"],
        default="all",
        help="生成フォーマット",
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # FastAPIアプリケーションの作成
    app = create_app()
    
    # OpenAPI仕様の取得
    openapi_spec = app.openapi()
    
    # OpenAPI JSONの保存
    if args.format in ["openapi", "all"]:
        openapi_path = output_dir / "openapi.json"
        with open(openapi_path, "w") as f:
            json.dump(openapi_spec, f, indent=2, ensure_ascii=False)
        print(f"Generated OpenAPI specification: {openapi_path}")
        
        # 人間が読みやすいMarkdown版も生成
        generate_markdown_docs(openapi_spec, output_dir / "api-reference.md")
        
    # Postmanコレクションの生成
    if args.format in ["postman", "all"]:
        postman_path = output_dir / "tsukuyomi-tts.postman_collection.json"
        generate_postman_collection(openapi_spec, postman_path)
        
    # Python SDKの生成
    if args.format in ["sdk-python", "all"]:
        sdk_dir = output_dir / "sdk" / "python"
        generate_client_sdk(openapi_spec, "python", sdk_dir)
        
        # setup.pyも生成
        setup_content = '''from setuptools import setup, find_packages

setup(
    name="tsukuyomi-tts-client",
    version="1.0.0",
    description="Tsukuyomi TTS API Client for Python",
    author="Tsukuyomi Team",
    packages=find_packages(),
    install_requires=[
        "requests>=2.28.0",
    ],
    python_requires=">=3.8",
)
'''
        with open(sdk_dir / "setup.py", "w") as f:
            f.write(setup_content)
            
    # TypeScript SDKの生成
    if args.format in ["sdk-typescript", "all"]:
        sdk_dir = output_dir / "sdk" / "typescript"
        generate_client_sdk(openapi_spec, "typescript", sdk_dir)
        
        # package.jsonも生成
        package_json = {
            "name": "tsukuyomi-tts-client",
            "version": "1.0.0",
            "description": "Tsukuyomi TTS API Client for TypeScript",
            "main": "dist/index.js",
            "types": "dist/index.d.ts",
            "scripts": {
                "build": "tsc",
                "test": "jest",
            },
            "keywords": ["tts", "text-to-speech", "tsukuyomi"],
            "license": "MIT",
            "devDependencies": {
                "@types/node": "^18.0.0",
                "typescript": "^5.0.0",
            },
        }
        
        with open(sdk_dir / "package.json", "w") as f:
            json.dump(package_json, f, indent=2)
            
        # tsconfig.jsonも生成
        tsconfig = {
            "compilerOptions": {
                "target": "ES2020",
                "module": "commonjs",
                "lib": ["ES2020"],
                "outDir": "./dist",
                "rootDir": "./",
                "strict": true,
                "esModuleInterop": true,
                "skipLibCheck": true,
                "forceConsistentCasingInFileNames": true,
                "declaration": true,
            },
        }
        
        with open(sdk_dir / "tsconfig.json", "w") as f:
            json.dump(tsconfig, f, indent=2)
            
    print(f"\n✅ APIドキュメントの生成が完了しました: {output_dir}")


def generate_markdown_docs(openapi_spec: dict, output_path: Path):
    """Markdown形式のAPIドキュメント生成"""
    md_content = f"""# Tsukuyomi TTS API リファレンス

{openapi_spec['info']['description']}

## ベースURL

- 本番環境: `https://api.tsukuyomi-tts.com`
- ステージング: `https://staging-api.tsukuyomi-tts.com`
- 開発環境: `http://localhost:8000`

## 認証

すべてのAPIリクエストには認証が必要です。以下のいずれかの方法を使用してください：

### APIキー認証
```
X-API-Key: YOUR_API_KEY
```

### Bearer認証 (JWT)
```
Authorization: Bearer YOUR_JWT_TOKEN
```

## エンドポイント

"""
    
    # エンドポイントの一覧
    for path, methods in openapi_spec.get("paths", {}).items():
        for method, operation in methods.items():
            if method in ["get", "post", "put", "delete", "patch"]:
                md_content += f"### {method.upper()} {path}\n\n"
                md_content += f"{operation.get('summary', '')}\n\n"
                
                if operation.get('description'):
                    md_content += f"{operation['description']}\n\n"
                    
                # パラメータ
                if operation.get('parameters'):
                    md_content += "**パラメータ:**\n\n"
                    for param in operation['parameters']:
                        required = "必須" if param.get('required') else "任意"
                        md_content += f"- `{param['name']}` ({param.get('schema', {}).get('type', 'string')}, {required}): {param.get('description', '')}\n"
                    md_content += "\n"
                    
                # リクエストボディ
                if operation.get('requestBody'):
                    md_content += "**リクエストボディ:**\n\n"
                    content = operation['requestBody'].get('content', {})
                    if 'application/json' in content:
                        schema = content['application/json'].get('schema', {})
                        if '$ref' in schema:
                            ref_name = schema['$ref'].split('/')[-1]
                            md_content += f"See [{ref_name}](#schemas) schema\n\n"
                        elif 'example' in schema:
                            md_content += "```json\n"
                            md_content += json.dumps(schema['example'], indent=2, ensure_ascii=False)
                            md_content += "\n```\n\n"
                            
                # レスポンス
                if operation.get('responses'):
                    md_content += "**レスポンス:**\n\n"
                    for status, response in operation['responses'].items():
                        md_content += f"- `{status}`: {response.get('description', '')}\n"
                    md_content += "\n"
                    
                md_content += "---\n\n"
                
    # スキーマ定義
    md_content += "## スキーマ定義 {#schemas}\n\n"
    
    for schema_name, schema in openapi_spec.get("components", {}).get("schemas", {}).items():
        md_content += f"### {schema_name}\n\n"
        
        if schema.get('description'):
            md_content += f"{schema['description']}\n\n"
            
        if schema.get('properties'):
            md_content += "| フィールド | 型 | 必須 | 説明 |\n"
            md_content += "|-----------|-----|------|------|\n"
            
            required_fields = schema.get('required', [])
            
            for prop_name, prop_schema in schema['properties'].items():
                prop_type = prop_schema.get('type', 'string')
                required = "✓" if prop_name in required_fields else ""
                description = prop_schema.get('description', '')
                
                # 配列の場合
                if prop_type == 'array':
                    items_type = prop_schema.get('items', {}).get('type', 'any')
                    prop_type = f"array<{items_type}>"
                    
                md_content += f"| {prop_name} | {prop_type} | {required} | {description} |\n"
                
        if 'example' in schema:
            md_content += "\n**例:**\n```json\n"
            md_content += json.dumps(schema['example'], indent=2, ensure_ascii=False)
            md_content += "\n```\n"
            
        md_content += "\n"
        
    # エラーコード
    md_content += """## エラーコード

| コード | 説明 | 対処法 |
|--------|------|--------|
| 400 | 不正なリクエスト | リクエストパラメータを確認してください |
| 401 | 認証エラー | APIキーまたはトークンを確認してください |
| 403 | アクセス拒否 | 権限を確認してください |
| 404 | リソースが見つかりません | URLとリソースIDを確認してください |
| 429 | レート制限超過 | しばらく待ってから再試行してください |
| 500 | サーバーエラー | サポートにお問い合わせください |

## SDKとツール

### 公式SDK

- [Python SDK](./sdk/python)
- [TypeScript SDK](./sdk/typescript)

### その他のツール

- [Postmanコレクション](./tsukuyomi-tts.postman_collection.json)
- [OpenAPI仕様](./openapi.json)

## サンプルコード

### Python
```python
from tsukuyomi_client import TsukuyomiClient, TTSRequest

client = TsukuyomiClient(api_key="YOUR_API_KEY")
request = TTSRequest(text="こんにちは、世界！", speaker_id=0)
result = client.synthesize(request)
print(f"Audio URL: {result['audio_url']}")
```

### TypeScript
```typescript
import { TsukuyomiClient } from 'tsukuyomi-tts-client';

const client = new TsukuyomiClient('YOUR_API_KEY');
const result = await client.synthesize({
  text: 'こんにちは、世界！',
  speakerId: 0,
});
console.log(`Audio URL: ${result.audioUrl}`);
```

### cURL
```bash
curl -X POST https://api.tsukuyomi-tts.com/v1/synthesis \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "こんにちは、世界！",
    "speaker_id": 0
  }'
```
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"Generated Markdown documentation: {output_path}")


if __name__ == "__main__":
    main()