#include "common.h"
#include "common-sdl.h"
#include "whisper.h"

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <iostream>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

//------------------------------------------------------------------------------

//  500 -> 00:05.000
// 6000 -> 01:00.000
std::string to_timestamp(int64_t t) {
    int64_t sec = t/100;
    int64_t msec = t - sec*100;
    int64_t min = sec/60;
    sec = sec - min*60;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d.%03d", (int) min, (int) sec, (int) msec);

    return std::string(buf);
}

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool speed_up      = false;
    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"   || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                 arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                 arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                 arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"   || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"  || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"  || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-vth" || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth" || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-su"  || arg == "--speed-up")      { params.speed_up      = true; }
        else if (arg == "-tr"  || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"  || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"  || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"  || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"   || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"   || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-f"   || arg == "--file")          { params.fname_out     = argv[++i]; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --step N        [%-7d] audio step size in milliseconds\n",                params.step_ms);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -su,      --speed-up      [%-7s] speed up audio by x2 (reduced accuracy)\n",        params.speed_up ? "true" : "false");
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between audio chunks\n",              params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n",                                params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -f FNAME, --file FNAME    [%-7s] text output file name\n",                          params.fname_out.c_str());
    fprintf(stderr, "\n");
}

std::array<int16_t, 256> mu_law_lookup;

void init_mu_law_lookup() {
    for (int i = 0; i < 256; i++) {
        int sgn = ((~i) & 0x80) ? -1 : 1;
        int exponent = (~i & 0x70) >> 4;
        int mantissa = ~i & 0x0F;
        int sample = sgn * (mantissa * 4 + 132) << (exponent - 1);
        mu_law_lookup[i] = sample;
    }
}

std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;

    for (unsigned int i = 0; i < hex.length(); i += 2) {
        std::string byteString = hex.substr(i, 2);
        uint8_t byte = (uint8_t) strtol(byteString.c_str(), NULL, 16);
        bytes.push_back(byte);
    }

    return bytes;
}

std::vector<int16_t> decode_mu_law(const std::vector<uint8_t>& input) {
    std::vector<int16_t> output;
    output.reserve(input.size());

    for (uint8_t sample : input) {
        output.push_back(mu_law_lookup[sample]);
    }

    return output;
}

std::vector<float> pcm16_to_float(const std::vector<int16_t>& input) {
    std::vector<float> output;
    output.reserve(input.size());

    for (int16_t sample : input) {
        // Scale the 16-bit PCM sample to the range [-1.0, 1.0]
        output.push_back(sample / 32768.0f);
    }

    return output;
}

std::vector<float> resample(const std::vector<float>& input, int input_rate, int output_rate) {
    std::vector<float> output;
    double rate = static_cast<double>(input_rate) / output_rate;
    for (double i = 0; i < input.size() - 1; i += rate) {
        int index = static_cast<int>(i);
        double fraction = i - index;
        float sample = static_cast<float>((1.0 - fraction) * input[index] + fraction * input[index + 1]);
        output.push_back(sample);
    }
    return output;
}

void write_word(std::ostream& stream, uint32_t value) {
    stream.put((value >> 0) & 0xFF);
    stream.put((value >> 8) & 0xFF);
    stream.put((value >> 16) & 0xFF);
    stream.put((value >> 24) & 0xFF);
}

void write_halfword(std::ostream& stream, uint16_t value) {
    stream.put((value >> 0) & 0xFF);
    stream.put((value >> 8) & 0xFF);
}

void write_wav(const std::vector<int16_t>& data, int sample_rate, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);

    // Write the RIFF chunk descriptor
    file << "RIFF";
    write_word(file, 36 + data.size() * sizeof(int16_t)); // ChunkSize
    file << "WAVE";

    // Write the "fmt " sub-chunk
    file << "fmt ";
    write_word(file, 16); // Subchunk1Size is 16
    write_halfword(file, 1); // PCM is format 1
    write_halfword(file, 1); // Mono
    write_word(file, sample_rate); // Sample rate
    write_word(file, sample_rate * sizeof(int16_t)); // ByteRate
    write_halfword(file, sizeof(int16_t)); // BlockAlign
    write_halfword(file, 8 * sizeof(int16_t)); // BitsPerSample

    // Write the "data" sub-chunk
    file << "data";
    write_word(file, data.size() * sizeof(int16_t));

    // Write the audio data
    for (int16_t sample : data) {
        write_halfword(file, sample);
    }
}

void do_session(tcp::socket& socket, const whisper_params& params, struct whisper_context * ctx)
{
    try
    {
        std::vector<int16_t> buffer;
        int last_sequence_number = 0;

        websocket::stream<tcp::socket> ws{std::move(socket)};
        ws.accept();

        for(;;)
        {
            beast::multi_buffer msg_buffer;
            ws.read(msg_buffer);

            auto message = beast::buffers_to_string(msg_buffer.data());

            auto j = nlohmann::json::parse(message);

            std::cout << "Received message with track: " << j["media"]["track"]
                      << " and timestamp: " << j["media"]["timestamp"] << std::endl;

            std::string audio_hex = j["media"]["payload"];
            std::vector<uint8_t> audio_bytes = hex_to_bytes(audio_hex);
            std::vector<int16_t> audio_pcm = decode_mu_law(audio_bytes);
            std::vector<float> audio_pcm_float = pcm16_to_float(audio_pcm);
            std::vector<float> audio_pcm_float_resampled = resample(audio_pcm_float, 8000, 16000);

            int sequence_number = std::stoi(j["sequenceNumber"].get<std::string>());

            if(sequence_number == last_sequence_number + 1) {
                buffer.insert(buffer.end(), audio_pcm.begin(), audio_pcm.end());
                last_sequence_number = sequence_number;
            } else {
                // Handle out-of-order or missing chunks
            }

            if(buffer.size() >= 3*16000) {

                // Create the whisper parameters
                whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
                wparams.print_progress   = false;
                wparams.print_special    = false;
                wparams.print_realtime   = false;
                wparams.print_timestamps = false;
                wparams.translate        = false;
                wparams.single_segment   = true;
                wparams.max_tokens       = 0;
                wparams.language         = "en";
                wparams.n_threads        = std::min(4, (int32_t) std::thread::hardware_concurrency());
                wparams.audio_ctx        = 0;
                wparams.speed_up         = false;


                // echo to screen that we are writing wav
                std::cout << "Writing wav" << std::endl;
                write_wav(buffer, 16000, "audio.wav");

                // Run the inference
                if (whisper_full(ctx, wparams, audio_pcm_float.data(), buffer.size()) != 0) {
                    fprintf(stderr, "failed to process audio\n");
                    return;
                }

                // Print result
                const int n_segments = whisper_full_n_segments(ctx);
                for (int i = 0; i < n_segments; ++i) {
                    const char * text = whisper_full_get_segment_text(ctx, i);
                    std::cout << "Segment " << i << ": " << text << std::endl;
                }

                // Clear buffer for next round of processing
                buffer.clear();

                // Destroy the whisper context
                whisper_free(ctx);

                ws.text(ws.got_text());
                ws.write(msg_buffer.data());
            }
        }
    }
    catch(beast::system_error const& se)
    {
        if(se.code() != websocket::error::closed)
            std::cerr << "Error: " << se.code().message() << std::endl;
    }
    catch(std::exception const& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}



//------------------------------------------------------------------------------


int main(int argc, char* argv[])
{
    // Initialize the mu-law lookup table
    init_mu_law_lookup();

    // Parse command-line parameters (from stream.cpp)
    whisper_params params;
    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    // Initialize the whisper context
    struct whisper_context * ctx = whisper_init_from_file(params.model.c_str());
    if (ctx == NULL) {
        fprintf(stderr, "failed to create whisper context\n");
        return 1;
    }

    try {
        auto const address = net::ip::make_address("0.0.0.0");
        auto const port = static_cast<unsigned short>(std::atoi("8080"));

        net::io_context ioc{1};

        tcp::acceptor acceptor{ioc, {address, port}};
        for(;;)
        {
            tcp::socket socket{ioc};

            acceptor.accept(socket);

            // pass params to do_session
            std::thread{std::bind(&do_session, std::move(socket), params, ctx)}.detach();
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    // Free the whisper context at the end
    whisper_free(ctx);

    return 0; // Return 0 for success
}
