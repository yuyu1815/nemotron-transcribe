# sherpa-onnx DirectML 実行体

`mise run setup` は `.runtime/` の実行体が不足・破損した場合に、[fork](https://github.com/yuyu1815/sherpa-onnx) の `9d68746e95905c7bf7b038f0505a737ff5187617` からビルドします。fork は上流 `079a6f05d05a18bc6e993704a557c2f62c7a17a5` を基に、Windows DirectML の依存を更新しています。

| 依存 | バージョン | パッケージ SHA-256 |
|---|---|---|
| Microsoft.ML.OnnxRuntime.DirectML | 1.24.4 | `57e9f11b73437bef7a309496135d4c1f96b1a8e9ddba60013fa27bfc1d788681` |
| Microsoft.AI.DirectML | 1.15.4 | `4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9` |

従来の 1.14.1 / 1.15.0 では、このモデルの DirectML 認識結果に反復が発生しました。fork の固定コミットは CMake で各 NuGet パッケージを取得し、上記ハッシュを検証します。2022 Build Tools / MSVC 19.44、Windows SDK 10.0.26100.0、CMake 4.0.0-rc4 のクリーンビルドと GPU 認識を確認しています。

セットアップは以下を行います。

- 正常な `.runtime/` は再ビルドしません。必要なときだけ一時ディレクトリでビルドし、DirectML の計算を確認してから配置します。
- `tools/dxgi.cpp` から GPU アダプター確認ツールをビルドします。DirectML デバイス 0 がソフトウェアではないことを確認します。
- 不足する CMake、Git、C++ Build Tools、Windows SDK は `winget` や Visual Studio Installer で追加します。管理者権限や再起動が必要な場合があります。
- AMD GPU ドライバーは自動インストールしません。対応版を別途準備してください。

DirectML は一部の処理を CPU に割り当てます。ROCm と DirectML のデバイス 0 が同じ物理 GPU とは限らないため、複数 GPU 環境では別途確認してください。
