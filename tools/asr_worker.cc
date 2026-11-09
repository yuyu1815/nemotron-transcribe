// Persistent, serial ReazonSpeech worker. No C API, model reload, or CPU retry.
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <string>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <io.h>
#include <process.h>

#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/wave-reader.h"

namespace {
constexpr size_t kMaxLine = 32768;
constexpr size_t kMaxBatchCount = 32;
constexpr size_t kMaxBatchPaddedSeconds = 60;

// Only errors explicitly classified at the input boundary are recoverable.
class InputError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

bool SafePath(const std::string &path) {
  if (path.empty() || std::any_of(path.begin(), path.end(), [](unsigned char c) {
        return c < 32 || c == 127;
      })) return false;
  try {
    // Windows' UTF-8 conversion rejects malformed byte sequences.
    return std::filesystem::u8path(path).is_absolute();
  } catch (const std::system_error &) {
    return false;
  }
}

bool ParseRequest(const std::string &line, uint64_t *id,
                  std::vector<std::string> *paths) {
  *id = 0;
  paths->clear();
  const auto tab = line.find('\t');
  if (line.size() > kMaxLine || tab == std::string::npos || tab == 0 ||
      line[0] < '1' || line[0] > '9') return false;
  auto parsed = std::from_chars(line.data(), line.data() + tab, *id);
  if (parsed.ec != std::errc() || parsed.ptr != line.data() + tab || *id == 0)
    return false;
  size_t start = tab + 1;
  while (true) {
    const auto end = line.find('\t', start);
    if (paths->size() == kMaxBatchCount) throw InputError("Too many WAV paths");
    auto path = line.substr(start, end == std::string::npos ? end : end - start);
    if (!SafePath(path)) throw InputError("Expected absolute UTF-8 WAV path");
    paths->push_back(std::move(path));
    if (end == std::string::npos) return true;
    start = end + 1;
  }
}

void CheckBatchBudget(size_t count, size_t sample_count) {
  // Per-file checking is equivalent to count * max(length), without rounding
  // or retaining every WAV. A legacy singleton deliberately has no new cap.
  if (count > 1 && sample_count > 16000 * kMaxBatchPaddedSeconds / count)
    throw InputError("Batch exceeds padded audio budget");
}

// Bounded before allocation; EOF is normal only between complete requests.
bool ReadLine(std::string *line) {
  line->clear();
  char c;
  while (std::cin.get(c)) {
    if (c == '\n') return true;
    if (line->size() == kMaxLine) throw std::runtime_error("Request too long");
    line->push_back(c);
  }
  if (!std::cin.eof() || !line->empty())
    throw std::runtime_error("Incomplete request or input failure");
  return false;
}

std::vector<float> ReadAudio(std::istream &input) {
  int32_t rate = 0;
  bool ok = false;
  auto samples = sherpa_onnx::ReadWave(input, &rate, &ok);
  if (!ok || rate != 16000 || samples.empty() ||
      samples.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
    throw InputError("Expected nonempty mono 16000 Hz WAV");
  // Preserve the existing minimum before measuring the batch budget.
  samples.resize(std::max<size_t>(samples.size(), 8000), 0.0f);
  return samples;
}

std::vector<float> ReadAudio(const std::string &path) {
  // Open via the UTF-8 filesystem conversion, not Windows' ANSI filename API.
  std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
  return ReadAudio(input);
}

std::string Decode(sherpa_onnx::OfflineRecognizer &recognizer,
                   const std::vector<float> &samples) {
  auto stream = recognizer.CreateStream();
  stream->AcceptWaveform(16000, samples.data(), static_cast<int32_t>(samples.size()));
  recognizer.DecodeStream(stream.get());
  return stream->GetResult().AsJsonString();
}

std::string DecodeRequest(sherpa_onnx::OfflineRecognizer &recognizer,
                          const std::vector<std::string> &paths) {
  std::vector<std::unique_ptr<sherpa_onnx::OfflineStream>> streams;
  std::vector<sherpa_onnx::OfflineStream *> pointers;
  for (const auto &path : paths) {
    auto samples = ReadAudio(path);
    CheckBatchBudget(paths.size(), samples.size());
    auto stream = recognizer.CreateStream();
    stream->AcceptWaveform(16000, samples.data(), static_cast<int32_t>(samples.size()));
    pointers.push_back(stream.get());
    streams.push_back(std::move(stream));
  }
  recognizer.DecodeStreams(pointers.data(), static_cast<int32_t>(pointers.size()));
  // DecodeStreams may reorder its pointers; ownership stays in request order.
  if (streams.size() == 1) return streams[0]->GetResult().AsJsonString();
  std::string result = "[";
  for (const auto &stream : streams) {
    if (result.size() > 1) result += ',';
    result += stream->GetResult().AsJsonString();
  }
  return result + ']';
}

void Error(uint64_t id, bool fatal) {
  std::cout << "{\"v\":1,\"id\":" << id
            << ",\"ok\":false,\"fatal\":" << (fatal ? "true" : "false")
            << ",\"error\":\"ASR worker failed; see stderr\"}" << std::endl;
}

void SelfTest() {
  uint64_t id = 0;
  std::vector<std::string> paths;
  if (!ParseRequest("1\tC:/audio/example.wav", &id, &paths) || id != 1 ||
      paths != std::vector<std::string>{"C:/audio/example.wav"})
    throw std::runtime_error("Valid single request rejected");
  const std::vector<std::string> ordered = {"C:/long.wav", u8"C:/\u77ed.wav", "C:/long.wav"};
  if (!ParseRequest(u8"18446744073709551615\tC:/long.wav\tC:/\u77ed.wav\tC:/long.wav",
                    &id, &paths) || id != UINT64_MAX || paths != ordered)
    throw std::runtime_error("Batch order, duplicate, UTF-8 or ID boundary failed");
  for (const auto &line : std::vector<std::string>{
           "", "1", "\tC:/a.wav", "0\tC:/a.wav", "01\tC:/a.wav",
           "-1\tC:/a.wav", "+1\tC:/a.wav", "18446744073709551616\tC:/a.wav",
           "1x\tC:/a.wav", "1\tC:/" + std::string(kMaxLine, 'a')}) {
    if (ParseRequest(line, &id, &paths))
      throw std::runtime_error("Malformed framing accepted");
  }
  for (const auto &line : std::vector<std::string>{
           "1\t", "1\trelative.wav", "1\tC:/a\tb.wav", "1\tC:/a.wav\t",
           "1\tC:/a.wav\t\tC:/b.wav", "1\tC:/a.wav\r", "1\tC:/a\n.wav",
           std::string("1\tC:/a\0.wav", 12), "1\tC:/a\x7f.wav",
           "1\tC:/\xff.wav", "1\tC:/\xc0\xaf.wav", "1\tC:/\xed\xa0\x80.wav"}) {
    try {
      ParseRequest(line, &id, &paths);
    } catch (const InputError &) {
      if (id != 1) throw std::runtime_error("Input error lost request ID");
      continue;
    }
    throw std::runtime_error("Invalid paths did not raise recoverable error");
  }
  std::string batch = "2";
  for (size_t i = 0; i < kMaxBatchCount; ++i) batch += "\tC:/a.wav";
  if (!ParseRequest(batch, &id, &paths) || paths.size() != kMaxBatchCount)
    throw std::runtime_error("Maximum batch rejected");
  try {
    ParseRequest(batch + "\tC:/b.wav", &id, &paths);
    throw std::runtime_error("Oversized batch accepted");
  } catch (const InputError &) {}
  const std::string exact = "3\tC:/" + std::string(kMaxLine - 5, 'a');
  if (!ParseRequest(exact, &id, &paths))
    throw std::runtime_error("Maximum line rejected");

  // One real PCM16 frame: use the native reader, not another WAV parser.
  const char wav[] = "RIFF\x26\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0"
                     "\x80\x3e\0\0\0\x7d\0\0\x02\0\x10\0data\x02\0\0\0\0\0";
  std::istringstream input(std::string(wav, sizeof(wav) - 1));
  const auto padded = ReadAudio(input);
  if (padded.size() != 8000 || padded.back() != 0.0f)
    throw std::runtime_error("Minimum 0.5 second padding failed");
  for (size_t count : {2, 4, 32}) {
    CheckBatchBudget(count, padded.size());
    const auto boundary = 16000 * kMaxBatchPaddedSeconds / count;
    CheckBatchBudget(count, boundary);
    try {
      CheckBatchBudget(count, boundary + 1);
      throw std::runtime_error("Over-budget batch accepted");
    } catch (const InputError &) {}
  }
  CheckBatchBudget(1, std::numeric_limits<int32_t>::max());
  std::istringstream bad_wav("not a WAV");
  try {
    ReadAudio(bad_wav);
    throw std::runtime_error("Bad WAV accepted");
  } catch (const InputError &) {}
}
}  // namespace

int wmain(int argc, wchar_t **wide_argv) {
  // Binary mode preserves CR/NUL for rejection and emits UTF-8 JSON unchanged.
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
  uint64_t id = 0;
  try {
    // Windows launches use UTF-16; sherpa paths and the wire protocol use UTF-8.
    std::vector<std::string> arguments;
    for (int i = 0; i < argc; ++i)
      arguments.push_back(std::filesystem::path(wide_argv[i]).u8string());
    std::vector<const char *> argv;
    for (const auto &argument : arguments) argv.push_back(argument.c_str());
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--help") {
        std::cerr << "asr-worker --encoder=PATH --decoder=PATH --joiner=PATH "
                     "--tokens=PATH --provider=directml:provider.conf "
                     "--warmup=ABSOLUTE_WAV [--num-threads=1]\n"
                     "stdin: positive-request-id TAB absolute-WAV-path [TAB path ...] LF\n"
                     "one path: result object; multiple paths: ordered result array\n"
                     "batch: max 32 paths, count * max(padded seconds) <= 60\n"
                     "single paths keep the legacy duration limit\n";
        return 0;
      }
    }
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
      SelfTest();
      return 0;
    }
    sherpa_onnx::OfflineRecognizerConfig config;
    config.model_config.model_type = "transducer";
    config.model_config.num_threads = 1;
    config.model_config.provider = "directml:provider.conf";
    std::string warmup;
    sherpa_onnx::ParseOptions options("Persistent serial DirectML ASR worker");
    config.Register(&options);
    options.Register("warmup", &warmup, "Absolute path to one-second warmup WAV");
    options.Read(argc, argv.data());
    if (options.NumArgs() != 0 || !SafePath(warmup) ||
        config.model_config.model_type != "transducer" ||
        config.model_config.num_threads != 1 ||
        config.model_config.provider != "directml:provider.conf" ||
        config.decoding_method != "greedy_search" || !config.Validate())
      throw std::runtime_error("Invalid worker configuration");

    auto samples = ReadAudio(warmup);
    if (samples.size() != 16000) throw std::runtime_error("Warmup must be one second");
    sherpa_onnx::OfflineRecognizer recognizer(config);
    Decode(recognizer, samples);
    samples.clear();
    auto profiles = recognizer.EndProfiling();
    if (profiles.size() != 3) throw std::runtime_error("Expected three profiles");
    for (size_t i = 0; i < profiles.size(); ++i) {
      const auto &name = profiles[i];
      if (name.rfind("ort-profile-", 0) != 0 || name.size() < 6 ||
          name.substr(name.size() - 5) != ".json" ||
          name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") != std::string::npos ||
          !std::filesystem::is_regular_file(name) || std::filesystem::file_size(name) == 0 ||
          std::find(profiles.begin(), profiles.begin() + i, name) != profiles.begin() + i)
        throw std::runtime_error("Invalid or duplicate profiling output");
    }
    // Parent checks real non-Memcpy DmlExecutionProvider Node events before use.
    std::cout << "{\"v\":1,\"ready\":true,\"service\":\"asr\",\"pid\":" << _getpid()
              << ",\"profiles\":[\"" << profiles[0] << "\",\"" << profiles[1]
              << "\",\"" << profiles[2] << "\"],\"batch\":{\"max_count\":"
              << kMaxBatchCount << ",\"max_padded_seconds\":"
              << kMaxBatchPaddedSeconds << "}}" << std::endl;
    std::string line;
    std::vector<std::string> paths;
    while (ReadLine(&line)) {
      id = 0;
      try {
        if (!ParseRequest(line, &id, &paths)) throw std::runtime_error("Malformed request");
        auto result = DecodeRequest(recognizer, paths);
        std::cout << "{\"v\":1,\"id\":" << id << ",\"ok\":true,\"result\":"
                  << result << "}" << std::endl;
      } catch (const InputError &e) {
        std::cerr << e.what() << '\n';
        Error(id, false);
      }
      id = 0;
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "asr-worker: " << e.what() << '\n';
    Error(id, true);
    return 1;
  }
}
