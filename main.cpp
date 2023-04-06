/*!
* \file main.cpp
* \brief Example script demonstrating the use of the MLPCpp library within C++
code.
* \author E.C.Bunschoten
* \version 1.1.0
*
* MLPCpp Project Website: https://github.com/EvertBunschoten/MLPCpp
*
* Copyright (c) 2023 Evert Bunschoten

* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.

* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#include <iostream>
#include <string>
#include <vector>
/*--- Include the look-up MLP class ---*/
#include "include/CLookUp_ANN.hpp"

using namespace std;

int main() {
  /* PREPROCESSING START */

  /* Step 1: Generate MLP collection */

  /*--- First specify an array of MLP input file names (preprocessing) ---*/
  string input_filenames[] = {
      "MLP_1.mlp",
      "MLP_2.mlp"}; /*!< String array containing MLP input file names. */
  unsigned short nMLPs = sizeof(input_filenames) / sizeof(string);

  /*--- Generate a collection of MLPs with the architectures described in the
   * input file(s) ---*/
  MLPToolbox::CLookUp_ANN ANN_test =
      MLPToolbox::CLookUp_ANN(nMLPs, input_filenames);

  /* Step 2: Input-Output mapping (preprocessing) */
  /* Define an input-output map for each look-up operation to be performed. */
  vector<string>
      input_names, /*!< Controlling variable names for the look-up operation. */
      output_names; /*!< Output variable names for the look-up operation */

  /*--- Size the controlling variable vector and fill in the variable names
   * (should correspond to the controlling variable names in any of the loaded
   * MLPs, but the order is irrelevant) ---*/
  input_names.resize(3);
  input_names[0] = "CV_1";
  input_names[1] = "CV_2";
  input_names[2] = "CV_3";

  /*--- Size the output variable vector and set the variable names ---*/
  output_names.resize(3);
  output_names[0] = "Output_2";
  output_names[1] = "Output_3";
  output_names[2] = "Output_6";

  /*--- Generate the input-output map and pair the loaded MLP's with the input
   * and output variables of the lookup operation ---*/
  MLPToolbox::CIOMap iomap = MLPToolbox::CIOMap(input_names, output_names);
  ANN_test.PairVariableswithMLPs(iomap);

  /*--- Optional: display network architecture information in the terminal ---*/
  ANN_test.DisplayNetworkInfo();

  /*--- Pepare input and output vectors for look-up operation ---*/
  vector<double> MLP_inputs;
  vector<double *> MLP_outputs;

  MLP_inputs.resize(input_names.size());
  MLP_outputs.resize(output_names.size());

  /*--- Set pointer to output variables ---*/
  double val_output_2, val_output_3, val_output_6;
  MLP_outputs[0] = &val_output_2;
  MLP_outputs[1] = &val_output_3;
  MLP_outputs[2] = &val_output_6;

  /* PREPROCESSING END */

  /* Step 3: Evaluate MLPs (in iterative process)*/

  double val_cv_1 = -0.575;
  double val_cv_2 = 0;
  double val_cv_3 = 0.0144;

  while (val_cv_1 < 0.0) {
    MLP_inputs[0] = val_cv_1;
    MLP_inputs[1] = val_cv_2;
    MLP_inputs[2] = val_cv_3;

    /*--- Call the PredictANN function to evaluate the relevant MLPs for the
     * look-up process specified through the input-output map and set the output
     * values. ---*/
    auto inside = ANN_test.PredictANN(&iomap, MLP_inputs, MLP_outputs);
    cout << val_cv_1 << ", " << val_output_2 << ", " << val_output_3 << ", "
         << val_output_6 << ", " << inside << endl;

    val_cv_1 += 0.01;
  }
}
