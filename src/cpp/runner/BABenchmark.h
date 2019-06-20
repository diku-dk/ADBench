#pragma once

#include  "Benchmark.h"

template <>
BAInput read_input_data<BAInput>(const std::string& input_file, const bool replicate_point);

template <>
unique_ptr<ITest<BAInput, BAOutput>> get_test<BAInput, BAOutput>(const ModuleLoader& module_loader);

template<>
void save_output_to_file<BAOutput>(const BAOutput& output, const string& output_prefix, const string& input_basename,
                                   const string& module_basename);
