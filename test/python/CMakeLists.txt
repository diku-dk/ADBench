find_package(Python3 COMPONENTS Interpreter)

if (${Python3_Interpreter_FOUND})
    add_subdirectory ("runner")
    add_subdirectory("modules")
else()
    message(WARNING "Python3 Interpreter not found")
endif()