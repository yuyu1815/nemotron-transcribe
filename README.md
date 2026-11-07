# 話者付き文字起こし

Windows の AMD GPU で ReazonSpeech による日本語音声認識と Nemotron-3-Diarization による話者分離を行います。

## 実行

```powershell
mise install
mise run setup
mise run transcribe "C:\path\video.mp4"
# または
mise run transcribe "https://www.youtube.com/watch?v=VIDEO_ID"
```

動画・音声ファイルまたは動画 URL を渡します。ファイルの場合は入力と同じ場所、URL の場合はプロジェクトのディレクトリに、2 ファイルを保存します（URL では動画 ID が名前になります）。既存の結果は上書きしません。

- `video.transcript.log`：時刻付き文字起こし
- `video.speakers.log`：話者ごとの開始・終了時刻

UTF-8 のテキスト形式で、各行は `[時:分:秒.ミリ秒 - 時:分:秒.ミリ秒] 内容` です。

## 準備と制限

- Windows 11 x64、対応 AMD GPU と AMD の ROCm 対応ドライバーが必要です。ドライバーは自動変更しません。Python と uv は mise で管理します。
- `mise run setup` は必要なモデルを固定リビジョンから取得し、不足・破損している DirectML 実行体を [sherpa-onnx の fork](https://github.com/yuyu1815/sherpa-onnx/tree/windows-directml-reazonspeech) の固定コミットからビルドします。既存の正常なモデル・実行体は再利用します。準備後に ROCm と DirectML を GPU で確認します。
- Git と `ffmpeg` がない場合、また実行体のビルドに必要な CMake・Visual Studio C++ Build Tools・Windows SDK がない場合は、Windows の `winget` 等を使い追加します。初回ビルドにはインターネット、数 GB の空き容量と時間が必要で、権限確認や再起動が必要になる場合があります。詳細は `docs/directml-build.md` を参照してください。
- ASR は DirectML と CPU の混在実行です。GPU が使われていない場合はエラーにします。DirectML は DXGI のデバイス 0、話者分離は ROCm のデバイス 0 を使います。複数 GPU の場合は各デバイスを別途確認してください。
- 先に Nemotron で発話区間を推定し、各区間の元の混合音声を ReazonSpeech に渡します。話者ログでは、連続する同じ話者の区間の間隔が 0.5 秒以下なら一続きとして表示します（別の話者が間にあれば結合しません）。文字起こしの区間は結合せず、元音声基準の時刻で照合します。短い間隔に発話があるとは限らず、これは見やすくするための補間です。話者分離が検出しなかった発話は文字起こしされません。
- 重複区間は各話者についてそれぞれ処理しますが、音源分離は行いません。同じ混合音声を複数回認識するため、相手の発話や重複したテキストが含まれる場合があります。
- 短い区間は DirectML の実行のため末尾に無音を追加して最低 0.5 秒にします。追加した無音に対応するトークンは除きますが、認識結果はパディングの影響を受ける場合があります。区間の時刻は変更しません。
- 区間は 30 秒を超えても分割しません。ReazonSpeech は約 30 秒までの入力を想定しており、長い区間では認識精度低下や GPU メモリ不足が起こり得ます。GPU メモリを抑えるため区間を 1 件ずつ処理しますが、長い区間ではメモリ不足を防げる保証はありません。
- 話者分離は全音声を一度に処理するため、音声が長いと GPU メモリ不足になる可能性があります。一時 WAV・区間 WAV・GPU プロファイル用の空き容量も必要です。

## テスト

```powershell
mise run test       # ロジックのチェック
```
