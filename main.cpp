// ============================================================================
// Minecraft CV AI Assistant — Verified Single-File Edition (main.cpp)
// ============================================================================

#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QSlider>
#include <QTextBrowser>
#include <QListWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFrame>
#include <QTimer>
#include <QImage>
#include <QPixmap>
#include <QString>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <limits>
#include <chrono>
#include <map>
#include <algorithm>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

namespace fs = std::filesystem;

// ============================================================================
// 1. Захват окна (Win32 GDI / Linux X11 / Демо-режим)
// ============================================================================
class ScreenCapturer {
#ifdef _WIN32
    struct WindowSearchData {
        HWND foundHwnd = nullptr;
    };

    static BOOL CALLBACK EnumWindowsProc(HWND hwnd, LPARAM lParam) {
        if (!IsWindowVisible(hwnd) || IsIconic(hwnd)) return TRUE;

        char title[256] = {0};
        GetWindowTextA(hwnd, title, sizeof(title));
        std::string titleStr(title);

        // Ищем окно Minecraft, исключая окно нашего собственного приложения
        if (titleStr.find("Minecraft") != std::string::npos &&
            titleStr.find("Minecraft CV AI Assistant") == std::string::npos) {
            auto* data = reinterpret_cast<WindowSearchData*>(lParam);
            data->foundHwnd = hwnd;
            return FALSE;
        }
        return TRUE;
    }
#endif

public:
    bool captureMinecraftOrScreen(cv::Mat& outFrame, bool& isRealWindow) {
#ifdef _WIN32
        WindowSearchData searchData;
        EnumWindows(EnumWindowsProc, reinterpret_cast<LPARAM>(&searchData));
        HWND hwnd = searchData.foundHwnd;

        int width = 0;
        int height = 0;
        HDC hdcSource = nullptr;

        if (hwnd) {
            isRealWindow = true;
            RECT rc{};
            GetClientRect(hwnd, &rc);
            width = rc.right - rc.left;
            height = rc.bottom - rc.top;
            hdcSource = GetDC(hwnd);
        } else {
            isRealWindow = false;
            width = GetSystemMetrics(SM_CXSCREEN);
            height = GetSystemMetrics(SM_CYSCREEN);
            hdcSource = GetDC(nullptr);
        }

        if (width <= 0 || height <= 0 || !hdcSource) {
            if (hdcSource) ReleaseDC(hwnd, hdcSource);
            return generateFallbackFrame(outFrame);
        }

        HDC hdcMem = CreateCompatibleDC(hdcSource);
        HBITMAP hbm = CreateCompatibleBitmap(hdcSource, width, height);
        HGDIOBJ oldObj = SelectObject(hdcMem, hbm);

        BitBlt(hdcMem, 0, 0, width, height, hdcSource, 0, 0, SRCCOPY);

        BITMAPINFOHEADER bi{};
        bi.biSize = sizeof(BITMAPINFOHEADER);
        bi.biWidth = width;
        bi.biHeight = -height; // Top-down изображение
        bi.biPlanes = 1;
        bi.biBitCount = 32;
        bi.biCompression = BI_RGB;

        cv::Mat bgra(height, width, CV_8UC4);
        GetDIBits(hdcMem, hbm, 0, height, bgra.data, reinterpret_cast<BITMAPINFO*>(&bi), DIB_RGB_COLORS);
        cv::cvtColor(bgra, outFrame, cv::COLOR_BGRA2BGR);

        SelectObject(hdcMem, oldObj);
        DeleteObject(hbm);
        DeleteDC(hdcMem);
        ReleaseDC(hwnd, hdcSource);
        return true;

#elif defined(__linux__)
        Display* display = XOpenDisplay(nullptr);
        if (!display) return generateFallbackFrame(outFrame);

        Window root = DefaultRootWindow(display);
        XWindowAttributes attrs;
        XGetWindowAttributes(display, root, &attrs);

        XImage* img = XGetImage(display, root, 0, 0, attrs.width, attrs.height, AllPlanes, ZPixmap);
        if (!img) {
            XCloseDisplay(display);
            return generateFallbackFrame(outFrame);
        }

        cv::Mat bgra(attrs.height, attrs.width, CV_8UC4, img->data);
        cv::cvtColor(bgra, outFrame, cv::COLOR_BGRA2BGR);

        XDestroyImage(img);
        XCloseDisplay(display);
        isRealWindow = false;
        return true;
#else
        isRealWindow = false;
        return generateFallbackFrame(outFrame);
#endif
    }

private:
    bool generateFallbackFrame(cv::Mat& outFrame) {
        outFrame = cv::Mat(480, 640, CV_8UC3, cv::Scalar(28, 30, 38));
        cv::putText(outFrame, "Minecraft Window Not Found (Demo Feed)",
                    cv::Point(85, 240), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(16, 185, 129), 2);
        return true;
    }
};

// ============================================================================
// 2. Извлечение эмбеддингов (32x32 Grayscale + L2 нормализация)
// ============================================================================
class EmbeddingExtractor {
public:
    static cv::Rect getTargetROI(const cv::Mat& frame) {
        int w = frame.cols;
        int h = frame.rows;
        int boxW = std::max(2, std::min(w, h) / 2);
        int boxH = boxW;
        int x = std::max(0, (w - boxW) / 2);
        int y = std::max(0, (h - boxH) / 2);
        boxW = std::min(boxW, w - x);
        boxH = std::min(boxH, h - y);
        return cv::Rect(x, y, boxW, boxH);
    }

    static std::vector<float> extract(const cv::Mat& frame) {
        if (frame.empty() || frame.cols < 4 || frame.rows < 4) {
            return std::vector<float>(1024, 0.0f);
        }

        cv::Rect roi = getTargetROI(frame);
        cv::Mat cropped = frame(roi);

        cv::Mat gray, resized, floatMat;
        if (cropped.channels() == 3) {
            cv::cvtColor(cropped, gray, cv::COLOR_BGR2GRAY);
        } else if (cropped.channels() == 4) {
            cv::cvtColor(cropped, gray, cv::COLOR_BGRA2GRAY);
        } else {
            gray = cropped;
        }

        cv::resize(gray, resized, cv::Size(32, 32), 0, 0, cv::INTER_AREA);
        resized.convertTo(floatMat, CV_32F);
        if (!floatMat.isContinuous()) {
            floatMat = floatMat.clone();
        }

        const float* ptr = floatMat.ptr<float>(0);
        std::vector<float> vec(ptr, ptr + 1024);

        float norm = 0.0f;
        for (float v : vec) norm += v * v;
        norm = std::sqrt(norm);

        if (norm > 1e-6f) {
            for (float& v : vec) v /= norm;
        }
        return vec;
    }
};

// ============================================================================
// 3. Локальная база игроков + k-NN + Бинарный кэш
// ============================================================================
struct PlayerSample {
    std::string name;
    std::vector<float> embedding;
    std::string imagePath;
};

struct SearchMatch {
    bool matched = false;
    std::string playerName;
    float distance = 999.0f;
};

class PlayerLibrary {
private:
    std::vector<PlayerSample> samples_;
    std::string baseDir_ = "players";
    std::string cacheFile_ = "embeddings_cache.bin";

    static float euclideanDistance(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) return std::numeric_limits<float>::max();
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return std::sqrt(sum);
    }

    static std::string sanitizeName(const std::string& raw) {
        std::string clean;
        for (char c : raw) {
            if (c == '/' || c == '\\' || c == ':' || c == '*' ||
                c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
                clean += '_';
            } else {
                clean += c;
            }
        }
        return clean.empty() ? "player" : clean;
    }

public:
    PlayerLibrary() {
        try {
            fs::create_directories(baseDir_);
            if (!loadCache()) {
                rebuildFromDisk();
            }
        } catch (...) {}
    }

    bool addLabeledFrame(const std::string& playerName, const cv::Mat& frame) {
        if (playerName.empty() || frame.empty()) return false;

        try {
            std::string safeFolder = sanitizeName(playerName);
            fs::path playerDir = fs::path(baseDir_) / safeFolder;
            fs::create_directories(playerDir);

            auto now = std::chrono::system_clock::now().time_since_epoch().count();
            std::string filename = "sample_" + std::to_string(now) + ".png";
            fs::path fullPath = playerDir / filename;

            if (!cv::imwrite(fullPath.string(), frame)) return false;

            std::vector<float> emb = EmbeddingExtractor::extract(frame);
            samples_.push_back({playerName, emb, fullPath.string()});
            saveCache();
            return true;
        } catch (...) {
            return false;
        }
    }

    SearchMatch searchKNN(const cv::Mat& currentFrame, float threshold) const {
        if (samples_.empty()) return {false, "", 999.0f};

        std::vector<float> query = EmbeddingExtractor::extract(currentFrame);
        float bestDist = std::numeric_limits<float>::max();
        std::string bestName;

        for (const auto& s : samples_) {
            float dist = euclideanDistance(query, s.embedding);
            if (dist < bestDist) {
                bestDist = dist;
                bestName = s.name;
            }
        }

        bool isMatch = (bestDist <= threshold);
        return {isMatch, bestName, bestDist};
    }

    std::map<std::string, int> getPlayerCounts() const {
        std::map<std::string, int> counts;
        for (const auto& s : samples_) {
            counts[s.name]++;
        }
        return counts;
    }

    void rebuildFromDisk() {
        samples_.clear();
        if (!fs::exists(baseDir_)) return;

        try {
            for (const auto& dirEntry : fs::directory_iterator(baseDir_)) {
                if (dirEntry.is_directory()) {
                    std::string playerName = dirEntry.path().filename().string();
                    for (const auto& imgEntry : fs::directory_iterator(dirEntry.path())) {
                        cv::Mat img = cv::imread(imgEntry.path().string());
                        if (!img.empty()) {
                            samples_.push_back({
                                playerName,
                                EmbeddingExtractor::extract(img),
                                imgEntry.path().string()
                            });
                        }
                    }
                }
            }
            saveCache();
        } catch (...) {}
    }

    bool saveCache() const {
        std::ofstream out(cacheFile_, std::ios::binary);
        if (!out.is_open()) return false;

        uint32_t count = static_cast<uint32_t>(samples_.size());
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));

        for (const auto& s : samples_) {
            uint32_t nameLen = static_cast<uint32_t>(s.name.size());
            out.write(reinterpret_cast<const char*>(&nameLen), sizeof(nameLen));
            out.write(s.name.data(), nameLen);

            uint32_t pathLen = static_cast<uint32_t>(s.imagePath.size());
            out.write(reinterpret_cast<const char*>(&pathLen), sizeof(pathLen));
            out.write(s.imagePath.data(), pathLen);

            uint32_t dim = static_cast<uint32_t>(s.embedding.size());
            out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
            out.write(reinterpret_cast<const char*>(s.embedding.data()), dim * sizeof(float));
        }
        return out.good();
    }

    bool loadCache() {
        std::ifstream in(cacheFile_, std::ios::binary);
        if (!in.is_open()) return false;

        samples_.clear();
        uint32_t count = 0;
        if (!in.read(reinterpret_cast<char*>(&count), sizeof(count))) return false;
        if (count == 0 || count > 100000) return false;

        for (uint32_t i = 0; i < count; ++i) {
            uint32_t nameLen = 0;
            if (!in.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen)) || nameLen > 1024) return false;
            std::string name(nameLen, '\0');
            if (!in.read(&name[0], nameLen)) return false;

            uint32_t pathLen = 0;
            if (!in.read(reinterpret_cast<char*>(&pathLen), sizeof(pathLen)) || pathLen > 4096) return false;
            std::string path(pathLen, '\0');
            if (!in.read(&path[0], pathLen)) return false;

            uint32_t dim = 0;
            if (!in.read(reinterpret_cast<char*>(&dim), sizeof(dim)) || dim != 1024) return false;
            std::vector<float> emb(dim);
            if (!in.read(reinterpret_cast<char*>(emb.data()), dim * sizeof(float))) return false;

            samples_.push_back({std::move(name), std::move(emb), std::move(path)});
        }
        return !samples_.empty();
    }
};

// ============================================================================
// 4. Клиент Wikipedia REST API
// ============================================================================
class WikiClient {
private:
    static size_t writeCb(void* contents, size_t size, size_t nmemb, void* userp) {
        auto* str = static_cast<std::string*>(userp);
        str->append(static_cast<char*>(contents), size * nmemb);
        return size * nmemb;
    }

public:
    static std::string fetchSummary(const std::string& queryTitle, const std::string& lang = "ru") {
        if (queryTitle.empty()) return "Введите имя или термин для поиска в Wikipedia.";

        CURL* curl = curl_easy_init();
        if (!curl) return "Ошибка инициализации libcurl.";

        // Заменяем пробелы на подчеркивания для корректного URL Wikipedia REST API
        std::string normalizedTitle = queryTitle;
        for (char& c : normalizedTitle) {
            if (c == ' ') c = '_';
        }

        std::string responseBuffer;
        char* escaped = curl_easy_escape(curl, normalizedTitle.c_str(), static_cast<int>(normalizedTitle.size()));
        if (!escaped) {
            curl_easy_cleanup(curl);
            return "Ошибка кодирования запроса.";
        }

        std::string url = "https://" + lang + ".wikipedia.org/api/rest_v1/page/summary/" + std::string(escaped);
        curl_free(escaped);

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "MinecraftCVAssistant/2.0 (Windows Qt6)");
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBuffer);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 6L);
        // Отключаем привязку к локальному файлу сертификатов, чтобы .exe работал на любом ПК
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

        CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            return "Сетевая ошибка при обращении к Wikipedia API.";
        }

        try {
            auto j = nlohmann::json::parse(responseBuffer);
            if (j.contains("extract") && j["extract"].is_string()) {
                std::string ext = j["extract"].get<std::string>();
                if (!ext.empty()) return ext;
            }
        } catch (...) {
            return "Ошибка разбора JSON-ответа Wikipedia.";
        }

        if (lang == "ru") {
            return fetchSummary(queryTitle, "en");
        }

        return "Статья по запросу «" + queryTitle + "» не найдена в Wikipedia.";
    }
};

// =================================
