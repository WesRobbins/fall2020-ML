cmake_minimum_required(VERSION 3.17)
project(proj1)

set(CMAKE_CXX_STANDARD 17)

add_executable(proj1 main.cpp obj/DataClass.cpp obj/DataClass.h readers/main_reader.h ML.cpp ML.h DataClass.cpp DataClass.h Reader.cpp Reader.h DataLine.cpp DataLine.h Algorithm.cpp Algorithm.h Classifications.cpp Classifications.h Classifier.cpp Classifier.h Evaluate.cpp Evaluate.h)