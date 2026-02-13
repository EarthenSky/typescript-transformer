cmake_minimum_required(VERSION 3.28)
project(infer)

set(CMAKE_C_STANDARD 99)

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
