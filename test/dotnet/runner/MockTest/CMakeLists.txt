project("ADBench_dotnet_Runner_MockTest")

add_custom_target(DotnetRunner_MockTest ALL
    COMMAND "dotnet" "build" "--configuration" "${AD_BUILD_TYPE}" "--output" "${CMAKE_CURRENT_BINARY_DIR}" "-p:IntermediateOutputPath=${CMAKE_CURRENT_BINARY_DIR}/obj/" "${CMAKE_CURRENT_LIST_DIR}/MockTest.csproj"
    )