cmake_minimum_required(VERSION 3.26)
project(infer)

set(CMAKE_C_STANDARD 99)

include(CheckCSourceCompiles)
check_c_source_compiles("
int main() {
    _Float16 x = 1.0;
    float y = (float)x;
    return 0;
}" HAS_FLOAT16_TYPE)

if (NOT HAS_FLOAT16_TYPE)
    message(FATAL_ERROR "Your compiler does not support _Float16")
endif ()

add_library(
    ${PROJECT_NAME} SHARED
    infer.c
)

set_target_properties(
    ${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".node"
)

# for cmake-js
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_JS_INC})
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})

# Include N-API wrappers from node_modules
execute_process(
    COMMAND node -p "require('node-addon-api').include"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE NODE_ADDON_API_DIR
)
string(REPLACE "\n" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})

# Node headers (adjust path if needed)
target_include_directories(${PROJECT_NAME} PRIVATE ${NODE_ADDON_API_DIR})
