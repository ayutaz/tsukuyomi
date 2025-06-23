# Contributing to Tsukuyomi

## ブランチ戦略

### ブランチ構成
- `main`: プロダクション用ブランチ（保護されています）
- `dev`: 開発用ブランチ（全ての機能はここにマージ）
- `feat/xxx`: 新機能開発用（devから分岐）
- `fix/xxx`: バグ修正用（devから分岐）

### ワークフロー

1. **新機能開発**
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feat/your-feature-name
   # 開発作業
   git push origin feat/your-feature-name
   # PRをdevブランチに作成
   ```

2. **バグ修正**
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b fix/issue-description
   # 修正作業
   git push origin fix/issue-description
   # PRをdevブランチに作成
   ```

3. **リリース**
   - devブランチが安定したら、mainへのPRを作成
   - レビューとテストの後、mainにマージ

## CI/CD

### 自動テスト
以下のブランチへのpush/PRで自動実行されます：
- `main`
- `dev`
- `feat/**`
- `fix/**`

### テスト内容
- ユニットテスト（全OS、Python 3.8-3.11）
- コードフォーマットチェック（black, ruff, isort）
- 型チェック（mypy）
- カバレッジレポート

## 開発環境のセットアップ

1. **uvのインストール**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **開発環境構築**
   ```bash
   make install-dev
   ```

3. **pre-commitフックの設定**
   ```bash
   pre-commit install
   ```

## コーディング規約

### Pythonスタイル
- Black（line-length: 88）
- isort（profile: black）
- ruff（E, F, I, W）

### コミットメッセージ
- `feat:` 新機能
- `fix:` バグ修正
- `docs:` ドキュメント
- `test:` テスト
- `refactor:` リファクタリング
- `style:` コードスタイル
- `chore:` その他

例：
```
feat: Add Japanese accent prediction with F0-BERT

- Implement label-free prosody prediction
- Add continuous F0 modeling
- Support unknown words and accent combinations
```

## テスト

### テスト実行
```bash
# 全テスト
make test

# 高速テスト（GPU/遅いテストを除外）
make test-fast

# 特定のテスト
uv run pytest tests/test_frontend.py -v
```

### テスト作成
- 新機能には必ずテストを追加
- カバレッジ80%以上を維持
- モックを活用して外部依存を最小化

## プルリクエスト

### PRチェックリスト
- [ ] テストが全て通る
- [ ] コードフォーマットが適用されている
- [ ] 新機能にはテストがある
- [ ] ドキュメントが更新されている
- [ ] コミットメッセージが規約に従っている

### レビュープロセス
1. devブランチへのPRを作成
2. CI/CDが全て通ることを確認
3. コードレビューを受ける
4. 承認後、devにマージ

## 質問・提案

- Issueで議論を開始
- Discussionsで技術的な相談
- 大きな変更は事前にIssueで相談