cmake_minimum_required(VERSION 3.16)
project(MinecraftCVAssistant LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 COMPONENTS Widgets Core Gui REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(MinecraftCVAssistant main.cpp)

target_link_libraries(MinecraftCVAssistant PRIVATE
    Qt6::Widgets
    Qt6::Core
    Qt6::Gui
    ${OpenCV_LIBS}
    CURL::libcurl
    nlohmann_json::nlohmann_json
    gdi32
    user32
)
