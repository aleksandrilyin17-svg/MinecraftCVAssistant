// ============================================================================
// Minecraft CV AI Assistant — Single-File Edition (main.cpp)
// Стек: C++17, Qt6 Widgets, OpenCV, libcurl, nlohmann::json, Win32 GDI / X11
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

#include <opencv2/opencv.hpp>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <chrono>
#include <map>

#ifdef _WIN32
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
public:
    bool captureMinecraftOrScreen(cv::Mat& outFrame, bool& isRealWindow) {
#ifdef _WIN32
        HWND hwnd = FindWindowA(nullptr, "Minecraft");
        if (!hwnd) {
            hwnd = GetDesktopWindow();
            isRealWindow = false;
        } else {
            isRealWindow = true;
        }

        RECT rc;
        GetClientRect(hwnd, &rc);
        int width = rc.right - rc.left;
        int height = rc.bottom - rc.top;

        if (width <= 0 || height <= 0) return generateFallbackFrame(outFrame);

        HDC hdcScreen = GetDC(hwnd);
        HDC hdcMem = CreateCompatibleDC(hdcScreen);
        HBITMAP hbm = CreateCompatibleBitmap(hdcScreen, width, height);
        SelectObject(hdcMem, hbm);

        BitBlt(hdcMem, 0, 0, width, height, hdcScreen, 0, 0, SRCCOPY);

        BITMAPINFOHEADER bi{};
        bi.biSize = sizeof(BITMAPINFOHEADER);
        bi.biWidth = width;
        bi.biHeight = -height; // Top-down
        bi.biPlanes = 1;
        bi.biBitCount = 32;
        bi.biCompression = BI_RGB;

        cv::Mat bgra(height, width, CV_8UC4);
        GetDIBits(hdcMem, hbm, 0, height, bgra.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);
        cv::cvtColor(bgra, outFrame, cv::COLOR_BGRA2BGR);

        DeleteObject(hbm);
        DeleteDC(hdcMem);
        ReleaseDC(hwnd, hdcScreen);
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
        int boxW = std::min(w, h) / 2;
        int boxH = boxW;
        return cv::Rect((w - boxW) / 2, (h - boxH) / 2, boxW, boxH);
    }

    static std::vector<float> extract(const cv::Mat& frame) {
        if (frame.empty()) return std::vector<float>(1024, 0.0f);

        cv::Rect roi = getTargetROI(frame);
        cv::Mat cropped = frame(roi);

        cv::Mat gray, resized;
        if (cropped.channels() == 3) {
            cv::cvtColor(cropped, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = cropped;
        }

        cv::resize(gray, resized, cv::Size(32, 32), 0, 0, cv::INTER_AREA);
        resized.convertTo(resized, CV_32F);

        std::vector<float> vec(resized.begin<float>(), resized.end<float>());

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

public:
    PlayerLibrary() {
        fs::create_directories(baseDir_);
        if (!loadCache()) {
            rebuildFromDisk();
        }
    }

    bool addLabeledFrame(const std::string& playerName, const cv::Mat& frame) {
        if (playerName.empty() || frame.empty()) return false;

        fs::path playerDir = fs::path(baseDir_) / playerName;
        fs::create_directories(playerDir);

        auto now = std::chrono::system_clock::now().time_since_epoch().count();
        std::string filename = "sample_" + std::to_string(now) + ".png";
        fs::path fullPath = playerDir / filename;

        if (!cv::imwrite(fullPath.string(), frame)) return false;

        std::vector<float> emb = EmbeddingExtractor::extract(frame);
        samples_.push_back({playerName, emb, fullPath.string()});
        saveCache();
        return true;
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
    }

    bool saveCache() const {
        std::ofstream out(cacheFile_, std::ios::binary);
        if (!out) return false;

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
        return true;
    }

    bool loadCache() {
        std::ifstream in(cacheFile_, std::ios::binary);
        if (!in) return false;

        samples_.clear();
        uint32_t count = 0;
        in.read(reinterpret_cast<char*>(&count), sizeof(count));

        for (uint32_t i = 0; i < count; ++i) {
            uint32_t nameLen = 0;
            in.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen));
            std::string name(nameLen, '\0');
            in.read(&name[0], nameLen);

            uint32_t pathLen = 0;
            in.read(reinterpret_cast<char*>(&pathLen), sizeof(pathLen));
            std::string path(pathLen, '\0');
            in.read(&path[0], pathLen);

            uint32_t dim = 0;
            in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            std::vector<float> emb(dim);
            in.read(reinterpret_cast<char*>(emb.data()), dim * sizeof(float));

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
        ((std::string*)userp)->append((char*)contents, size * nmemb);
        return size * nmemb;
    }

public:
    static std::string fetchSummary(const std::string& queryTitle, const std::string& lang = "ru") {
        if (queryTitle.empty()) return "Введите имя или термин для поиска в Wikipedia.";

        CURL* curl = curl_easy_init();
        if (!curl) return "Ошибка инициализации libcurl.";

        std::string responseBuffer;
        char* escaped = curl_easy_escape(curl, queryTitle.c_str(), static_cast<int>(queryTitle.size()));
        std::string url = "https://" + lang + ".wikipedia.org/api/rest_v1/page/summary/" + std::string(escaped);
        curl_free(escaped);

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "MinecraftCVAssistant/2.0 (SingleFile Qt6)");
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBuffer);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

        CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            return "Сетевая ошибка при обращении к Wikipedia API.";
        }

        try {
            auto j = nlohmann::json::parse(responseBuffer);
            if (j.contains("extract") && !j["extract"].get<std::string>().empty()) {
                return j["extract"].get<std::string>();
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

// ============================================================================
// 5. Графический интерфейс (Qt6 MainWindow без необходимости MOC)
// ============================================================================
class MainWindow : public QMainWindow {
private:
    ScreenCapturer capturer_;
    PlayerLibrary library_;
    cv::Mat currentFrame_;
    bool isFrozen_ = false;
    float threshold_ = 0.35f;

    QLabel* videoLabel_;
    QLabel* statusBadge_;
    QPushButton* freezeBtn_;
    QLineEdit* addNameEdit_;
    QPushButton* addSampleBtn_;
    QLineEdit* queryNameEdit_;
    QSlider* thresholdSlider_;
    QLabel* thresholdValLabel_;
    QPushButton* identifyBtn_;
    QListWidget* libraryListWidget_;
    QTextBrowser* resultBrowser_;
    QTimer* captureTimer_;

public:
    explicit MainWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
        setWindowTitle("Minecraft CV AI Assistant — Single File Edition");
        resize(1180, 720);
        applyModernDarkStyle();
        setupUI();
        refreshLibraryList();

        captureTimer_ = new QTimer(this);
        connect(captureTimer_, &QTimer::timeout, this, [this]() { onTickFrame(); });
        captureTimer_->start(33);
    }

private:
    void onTickFrame() {
        if (isFrozen_) return;

        bool isRealMC = false;
        if (!capturer_.captureMinecraftOrScreen(currentFrame_, isRealMC)) return;

        cv::Mat displayMat = currentFrame_.clone();
        cv::Rect roi = EmbeddingExtractor::getTargetROI(displayMat);
        cv::rectangle(displayMat, roi, cv::Scalar(16, 185, 129), 2);
        cv::putText(displayMat, "SCAN ZONE (ROI)", cv::Point(roi.x + 8, roi.y + 24),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(16, 185, 129), 2);

        statusBadge_->setText(isRealMC
            ? "🟢 Захват: Окно Minecraft активно"
            : "🟡 Захват: Рабочий стол (Окно Minecraft не найдено)");

        cv::Mat rgb;
        cv::cvtColor(displayMat, rgb, cv::COLOR_BGR2RGB);
        QImage qimg(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
        videoLabel_->setPixmap(QPixmap::fromImage(qimg).scaled(
            videoLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    void onToggleFreeze() {
        isFrozen_ = !isFrozen_;
        if (isFrozen_) {
            freezeBtn_->setText("▶ Возобновить поток");
            freezeBtn_->setStyleSheet("background-color: #F59E0B; color: #111827; font-weight: bold;");
            statusBadge_->setText("⏸ Кадр заморожен — можно добавить игрока или распознать");
        } else {
            freezeBtn_->setText("⏸ Заморозить кадр (Freeze)");
            freezeBtn_->setStyleSheet("");
        }
    }

    void onAddPlayerSample() {
        QString qName = addNameEdit_->text().trimmed();
        if (qName.isEmpty()) {
            showResultCard("⚠️ Укажите ник игрока",
                           "Введите имя игрока в поле выше, чтобы сохранить текущий кадр в локальную библиотеку.",
                           "#F59E0B");
            return;
        }

        if (currentFrame_.empty()) return;

        if (library_.addLabeledFrame(qName.toStdString(), currentFrame_)) {
            refreshLibraryList();
            showResultCard("✅ Кадр добавлен в базу",
                           QString("Образец для игрока <b>%1</b> сохранен в <code>players/%1</code> и закэширован.").arg(qName),
                           "#10B981");
            addNameEdit_->clear();
        }
    }

    void onIdentifyOrAskWiki() {
        if (currentFrame_.empty()) return;

        QString requestedName = queryNameEdit_->text().trimmed();
        SearchMatch match = library_.searchKNN(currentFrame_, threshold_);

        if (match.matched) {
            QString html = QString(
                "<div style='color:#10B981; font-size:16px; font-weight:bold; margin-bottom:6px;'>"
                "🎯 Игрок распознан в локальной базе (k-NN)</div>"
                "<p style='font-size:15px;'>Никнейм: <b>%1</b></p>"
                "<p style='color:#9CA3AF;'>Евклидова дистанция: <b>%2</b> (Порог: %3)</p>"
            ).arg(QString::fromStdString(match.playerName))
             .arg(match.distance, 0, 'f', 3)
             .arg(threshold_, 0, 'f', 2);

            resultBrowser_->setHtml(html);
        } else {
            std::string wikiQuery = requestedName.isEmpty() ? match.playerName : requestedName.toStdString();
            if (wikiQuery.empty()) wikiQuery = "Minecraft";

            std::string wikiExtract = WikiClient::fetchSummary(wikiQuery);

            QString html = QString(
                "<div style='color:#60A5FA; font-size:16px; font-weight:bold; margin-bottom:6px;'>"
                "🌐 В локальной базе не найден — Справка из Wikipedia</div>"
                "<p style='color:#9CA3AF; font-size:12px;'>Ближайшая дистанция k-NN: %1 (выше порога %2) | Запрос: <b>%3</b></p>"
                "<hr style='border:1px solid #2D313E;'/>"
                "<p style='font-size:14px; line-height:1.5; color:#E5E7EB;'>%4</p>"
            ).arg(match.distance, 0, 'f', 3)
             .arg(threshold_, 0, 'f', 2)
             .arg(QString::fromStdString(wikiQuery))
             .arg(QString::fromStdString(wikiExtract));

            resultBrowser_->setHtml(html);
        }
    }

    void setupUI() {
        QWidget* central = new QWidget(this);
        setCentralWidget(central);

        QHBoxLayout* rootLayout = new QHBoxLayout(central);
        rootLayout->setContentsMargins(16, 16, 16, 16);
        rootLayout->setSpacing(16);

        QFrame* leftPanel = new QFrame();
        leftPanel->setObjectName("CardPanel");
        QVBoxLayout* leftLayout = new QVBoxLayout(leftPanel);
        leftLayout->setContentsMargins(14, 14, 14, 14);
        leftLayout->setSpacing(10);

        QHBoxLayout* topBar = new QHBoxLayout();
        statusBadge_ = new QLabel("🟡 Инициализация захвата...");
        statusBadge_->setStyleSheet("font-weight: 600; color: #D1D5DB;");
        freezeBtn_ = new QPushButton("⏸ Заморозить кадр (Freeze)");
        connect(freezeBtn_, &QPushButton::clicked, this, [this]() { onToggleFreeze(); });

        topBar->addWidget(statusBadge_);
        topBar->addStretch();
        topBar->addWidget(freezeBtn_);

        videoLabel_ = new QLabel();
        videoLabel_->setMinimumSize(640, 480);
        videoLabel_->setAlignment(Qt::AlignCenter);
        videoLabel_->setStyleSheet("background-color: #0B0C10; border-radius: 10px; border: 1px solid #262936;");

        leftLayout->addLayout(topBar);
        leftLayout->addWidget(videoLabel_, 1);

        QVBoxLayout* rightLayout = new QVBoxLayout();
        rightLayout->setSpacing(12);

        QFrame* trainCard = new QFrame();
        trainCard->setObjectName("CardPanel");
        QVBoxLayout* trainLayout = new QVBoxLayout(trainCard);
        QLabel* trainTitle = new QLabel("➕ Пополнить библиотеку игроков");
        trainTitle->setObjectName("SectionTitle");

        QHBoxLayout* trainInputRow = new QHBoxLayout();
        addNameEdit_ = new QLineEdit();
        addNameEdit_->setPlaceholderText("Ник игрока на экране (например, Notch)...");
        addSampleBtn_ = new QPushButton("Сохранить кадр");
        addSampleBtn_->setObjectName("PrimaryBtn");
        connect(addSampleBtn_, &QPushButton::clicked, this, [this]() { onAddPlayerSample(); });

        trainInputRow->addWidget(addNameEdit_, 1);
        trainInputRow->addWidget(addSampleBtn_);
        trainLayout->addWidget(trainTitle);
        trainLayout->addLayout(trainInputRow);

        QFrame* searchCard = new QFrame();
        searchCard->setObjectName("CardPanel");
        QVBoxLayout* searchLayout = new QVBoxLayout(searchCard);
        QLabel* searchTitle = new QLabel("🔍 Распознавание k-NN + Wikipedia API");
        searchTitle->setObjectName("SectionTitle");

        queryNameEdit_ = new QLineEdit();
        queryNameEdit_->setPlaceholderText("О ком спросить, если не распознан? (необязательно)");

        QHBoxLayout* sliderRow = new QHBoxLayout();
        QLabel* sliderText = new QLabel("Порог k-NN:");
        thresholdSlider_ = new QSlider(Qt::Horizontal);
        thresholdSlider_->setRange(5, 100);
        thresholdSlider_->setValue(35);
        thresholdValLabel_ = new QLabel("0.35");
        thresholdValLabel_->setFixedWidth(40);

        connect(thresholdSlider_, &QSlider::valueChanged, this, [this](int val) {
            threshold_ = val / 100.0f;
            thresholdValLabel_->setText(QString::number(threshold_, 'f', 2));
        });

        sliderRow->addWidget(sliderText);
        sliderRow->addWidget(thresholdSlider_, 1);
        sliderRow->addWidget(thresholdValLabel_);

        identifyBtn_ = new QPushButton("⚡ Распознать кадр / Найти в Wikipedia");
        identifyBtn_->setObjectName("AccentBtn");
        connect(identifyBtn_, &QPushButton::clicked, this, [this]() { onIdentifyOrAskWiki(); });

        searchLayout->addWidget(searchTitle);
        searchLayout->addWidget(queryNameEdit_);
        searchLayout->addLayout(sliderRow);
        searchLayout->addWidget(identifyBtn_);

        QFrame* libCard = new QFrame();
        libCard->setObjectName("CardPanel");
        QVBoxLayout* libLayout = new QVBoxLayout(libCard);
        QLabel* libTitle = new QLabel("📂 Локальная база эмбеддингов (Кэш)");
        libTitle->setObjectName("SectionTitle");
        libraryListWidget_ = new QListWidget();
        libraryListWidget_->setMaximumHeight(130);
        libLayout->addWidget(libTitle);
        libLayout->addWidget(libraryListWidget_);

        QFrame* resCard = new QFrame();
        resCard->setObjectName("CardPanel");
        QVBoxLayout* resLayout = new QVBoxLayout(resCard);
        QLabel* resTitle = new QLabel("💬 Ответ ИИ-ассистента");
        resTitle->setObjectName("SectionTitle");
        resultBrowser_ = new QTextBrowser();
        resultBrowser_->setOpenExternalLinks(true);
        showResultCard("Готов к работе",
                       "Наведите центральную рамку <b>SCAN ZONE</b> на игрока и нажмите «Распознать кадр», либо добавьте новый образец.",
                       "#9CA3AF");
        resLayout->addWidget(resTitle);
        resLayout->addWidget(resultBrowser_, 1);

        rightLayout->addWidget(trainCard);
        rightLayout->addWidget(searchCard);
        rightLayout->addWidget(libCard);
        rightLayout->addWidget(resCard, 1);

        rootLayout->addWidget(leftPanel, 3);
        rootLayout->addLayout(rightLayout, 2);
    }

    void refreshLibraryList() {
        libraryListWidget_->clear();
        auto counts = library_.getPlayerCounts();
        if (counts.empty()) {
            libraryListWidget_->addItem("База пуста — добавьте первый скриншот игрока");
            return;
        }
        for (const auto& [name, count] : counts) {
            libraryListWidget_->addItem(
                QString("👤 %1  —  %2 кадр(ов) в кэше").arg(QString::fromStdString(name)).arg(count)
            );
        }
    }

    void showResultCard(const QString& title, const QString& body, const QString& colorHex) {
        resultBrowser_->setHtml(QString(
            "<div style='color:%1; font-size:15px; font-weight:bold; margin-bottom:6px;'>%2</div>"
            "<div style='color:#D1D5DB; font-size:13px; line-height:1.45;'>%3</div>"
        ).arg(colorHex, title, body));
    }

    void applyModernDarkStyle() {
        setStyleSheet(R"(
            QMainWindow { background-color: #0F1117; }
            QWidget { color: #F3F4F6; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
            QFrame#CardPanel { background-color: #181B24; border: 1px solid #262A38; border-radius: 12px; }
            QLabel#SectionTitle { font-size: 14px; font-weight: bold; color: #FFFFFF; margin-bottom: 4px; }
            QLineEdit { background-color: #101218; border: 1px solid #2E3346; border-radius: 8px; padding: 8px 10px; color: #FFFFFF; }
            QLineEdit:focus { border: 1px solid #10B981; }
            QPushButton { background-color: #262B3D; border: 1px solid #353C52; border-radius: 8px; padding: 8px 14px; font-weight: 600; color: #E5E7EB; }
            QPushButton:hover { background-color: #31374D; }
            QPushButton#PrimaryBtn { background-color: #10B981; border: none; color: #06281E; font-weight: bold; }
            QPushButton#PrimaryBtn:hover { background-color: #34D399; }
            QPushButton#AccentBtn { background-color: #6366F1; border: none; color: #FFFFFF; padding: 10px; font-size: 14px; font-weight: bold; }
            QPushButton#AccentBtn:hover { background-color: #818CF8; }
            QListWidget, QTextBrowser { background-color: #101218; border: 1px solid #262A38; border-radius: 8px; padding: 6px; }
            QSlider::groove:horizontal { height: 6px; background: #262A38; border-radius: 3px; }
            QSlider::handle:horizontal { background: #10B981; width: 16px; margin: -5px 0; border-radius: 8px; }
        )");
    }
};

// ============================================================================
// 6. Точка входа
// ============================================================================
int main(int argc, char *argv[]) {
    curl_global_init(CURL_GLOBAL_DEFAULT);

    QApplication app(argc, argv);
    MainWindow window;
    window.show();

    int ret = app.exec();
    curl_global_cleanup();
    return ret;
}
