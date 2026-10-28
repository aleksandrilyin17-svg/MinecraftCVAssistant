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
- name: Configure
  shell: msys2 {0}
  run: |
    cmake -S . -B build -G "Ninja" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/ucrt64

- name: Build
  shell: msys2 {0}
  run: |
    cmake --build build --config Release
find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)

add_executable(MinecraftCVAssistant main.cpp)
target_link_libraries(MinecraftCVAssistant PRIVATE Qt6::Widgets ${OpenCV_LIBS} CURL::libcurl)
