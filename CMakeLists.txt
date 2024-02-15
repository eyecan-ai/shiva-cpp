cmake_minimum_required(VERSION 3.14)

project(shiva-cpp VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 --coverage -fkeep-inline-functions")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")

include_directories("include")

include(FetchContent)

FetchContent_Declare(
    json 
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

if(BUILD_TESTING)

    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/b3a9ba2b8e975550799838332803d468797ae2e1.zip
    )
    # for Windows: prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    file(GLOB TESTS_SRC tests/*.cpp)
    add_executable(tests ${TESTS_SRC})
    target_link_libraries(
        tests PRIVATE
        GTest::gtest
        GTest::gmock
        nlohmann_json::nlohmann_json
    )

    include(GoogleTest)
    gtest_discover_tests(tests)

    find_program(LCOV lcov REQUIRED)
    find_program(GENHTML genhtml REQUIRED)

    add_custom_target(coverage
        COMMAND ${LCOV} -d . -b ${CMAKE_SOURCE_DIR}/src --no-external --exclude ${CMAKE_BINARY_DIR}"/_deps/*" -c -o coverage.info
        COMMAND ${GENHTML} --demangle-cpp -o coverage coverage.info
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

else()

    add_executable(simple-client examples/simple_client.cpp)

    target_link_libraries(
        simple-client
        PRIVATE nlohmann_json::nlohmann_json
    )

endif(BUILD_TESTING)

include(FeatureSummary)
feature_summary(WHAT ALL)
