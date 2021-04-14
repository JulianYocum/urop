#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <tuple>

using namespace std;

vector<vector<double>> COORDS = {
{ -176.7, 44.1, -348.0 },
{ -176.7, 44.1, -290.0 },
{ -176.7, 44.1, -232.0 },
{ -176.7, 44.1, -174.0 },
{ -176.7, 44.1, -116.0 },
{ -176.7, 44.1, -58.0 },
{ -176.7, 44.1, 0.0 },
{ -176.7, 44.1, 58.0 },
{ -176.7, 44.1, 116.0 },
{ -176.7, 44.1, 174.0 },
{ -176.7, 44.1, 232.0 },
{ -176.7, 44.1, 290.0 },
{ -176.7, 44.1, 348.0 },
{ -117.3, 44.1, -348.0 },
{ -117.3, 44.1, -290.0 },
{ -117.3, 44.1, -232.0 },
{ -117.3, 44.1, -174.0 },
{ -117.3, 44.1, -116.0 },
{ -117.3, 44.1, -58.0 },
{ -117.3, 44.1, 0.0 },
{ -117.3, 44.1, 58.0 },
{ -117.3, 44.1, 116.0 },
{ -117.3, 44.1, 174.0 },
{ -117.3, 44.1, 232.0 },
{ -117.3, 44.1, 290.0 },
{ -117.3, 44.1, 348.0 },
{ -117.3, 102.9, -348.0 },
{ -117.3, 102.9, -290.0 },
{ -117.3, 102.9, -232.0 },
{ -117.3, 102.9, -174.0 },
{ -117.3, 102.9, -116.0 },
{ -117.3, 102.9, -58.0 },
{ -117.3, 102.9, 0.0 },
{ -117.3, 102.9, 58.0 },
{ -117.3, 102.9, 116.0 },
{ -117.3, 102.9, 174.0 },
{ -117.3, 102.9, 232.0 },
{ -117.3, 102.9, 290.0 },
{ -117.3, 102.9, 348.0 },
{ -176.7, 102.9, -348.0 },
{ -176.7, 102.9, -290.0 },
{ -176.7, 102.9, -232.0 },
{ -176.7, 102.9, -174.0 },
{ -176.7, 102.9, -116.0 },
{ -176.7, 102.9, -58.0 },
{ -176.7, 102.9, 0.0 },
{ -176.7, 102.9, 58.0 },
{ -176.7, 102.9, 116.0 },
{ -176.7, 102.9, 174.0 },
{ -176.7, 102.9, 232.0 },
{ -176.7, 102.9, 290.0 },
{ -176.7, 102.9, 348.0 },
{ -176.7, 191.1, -348.0 },
{ -176.7, 191.1, -290.0 },
{ -176.7, 191.1, -232.0 },
{ -176.7, 191.1, -174.0 },
{ -176.7, 191.1, -116.0 },
{ -176.7, 191.1, -58.0 },
{ -176.7, 191.1, 0.0 },
{ -176.7, 191.1, 58.0 },
{ -176.7, 191.1, 116.0 },
{ -176.7, 191.1, 174.0 },
{ -176.7, 191.1, 232.0 },
{ -176.7, 191.1, 290.0 },
{ -176.7, 191.1, 348.0 },
{ -117.3, 191.1, -348.0 },
{ -117.3, 191.1, -290.0 },
{ -117.3, 191.1, -232.0 },
{ -117.3, 191.1, -174.0 },
{ -117.3, 191.1, -116.0 },
{ -117.3, 191.1, -58.0 },
{ -117.3, 191.1, 0.0 },
{ -117.3, 191.1, 58.0 },
{ -117.3, 191.1, 116.0 },
{ -117.3, 191.1, 174.0 },
{ -117.3, 191.1, 232.0 },
{ -117.3, 191.1, 290.0 },
{ -117.3, 191.1, 348.0 },
{ -117.3, 249.9, -348.0 },
{ -117.3, 249.9, -290.0 },
{ -117.3, 249.9, -232.0 },
{ -117.3, 249.9, -174.0 },
{ -117.3, 249.9, -116.0 },
{ -117.3, 249.9, -58.0 },
{ -117.3, 249.9, 0.0 },
{ -117.3, 249.9, 58.0 },
{ -117.3, 249.9, 116.0 },
{ -117.3, 249.9, 174.0 },
{ -117.3, 249.9, 232.0 },
{ -117.3, 249.9, 290.0 },
{ -117.3, 249.9, 348.0 },
{ -176.7, 249.9, -348.0 },
{ -176.7, 249.9, -290.0 },
{ -176.7, 249.9, -232.0 },
{ -176.7, 249.9, -174.0 },
{ -176.7, 249.9, -116.0 },
{ -176.7, 249.9, -58.0 },
{ -176.7, 249.9, 0.0 },
{ -176.7, 249.9, 58.0 },
{ -176.7, 249.9, 116.0 },
{ -176.7, 249.9, 174.0 },
{ -176.7, 249.9, 232.0 },
{ -176.7, 249.9, 290.0 },
{ -176.7, 249.9, 348.0 },
{ -323.7, -29.4, -348.0 },
{ -323.7, -29.4, -290.0 },
{ -323.7, -29.4, -232.0 },
{ -323.7, -29.4, -174.0 },
{ -323.7, -29.4, -116.0 },
{ -323.7, -29.4, -58.0 },
{ -323.7, -29.4, 0.0 },
{ -323.7, -29.4, 58.0 },
{ -323.7, -29.4, 116.0 },
{ -323.7, -29.4, 174.0 },
{ -323.7, -29.4, 232.0 },
{ -323.7, -29.4, 290.0 },
{ -323.7, -29.4, 348.0 },
{ -264.3, -29.4, -348.0 },
{ -264.3, -29.4, -290.0 },
{ -264.3, -29.4, -232.0 },
{ -264.3, -29.4, -174.0 },
{ -264.3, -29.4, -116.0 },
{ -264.3, -29.4, -58.0 },
{ -264.3, -29.4, 0.0 },
{ -264.3, -29.4, 58.0 },
{ -264.3, -29.4, 116.0 },
{ -264.3, -29.4, 174.0 },
{ -264.3, -29.4, 232.0 },
{ -264.3, -29.4, 290.0 },
{ -264.3, -29.4, 348.0 },
{ -264.3, 29.4, -348.0 },
{ -264.3, 29.4, -290.0 },
{ -264.3, 29.4, -232.0 },
{ -264.3, 29.4, -174.0 },
{ -264.3, 29.4, -116.0 },
{ -264.3, 29.4, -58.0 },
{ -264.3, 29.4, 0.0 },
{ -264.3, 29.4, 58.0 },
{ -264.3, 29.4, 116.0 },
{ -264.3, 29.4, 174.0 },
{ -264.3, 29.4, 232.0 },
{ -264.3, 29.4, 290.0 },
{ -264.3, 29.4, 348.0 },
{ -323.7, 29.4, -348.0 },
{ -323.7, 29.4, -290.0 },
{ -323.7, 29.4, -232.0 },
{ -323.7, 29.4, -174.0 },
{ -323.7, 29.4, -116.0 },
{ -323.7, 29.4, -58.0 },
{ -323.7, 29.4, 0.0 },
{ -323.7, 29.4, 58.0 },
{ -323.7, 29.4, 116.0 },
{ -323.7, 29.4, 174.0 },
{ -323.7, 29.4, 232.0 },
{ -323.7, 29.4, 290.0 },
{ -323.7, 29.4, 348.0 },
{ -323.7, 117.6, -348.0 },
{ -323.7, 117.6, -290.0 },
{ -323.7, 117.6, -232.0 },
{ -323.7, 117.6, -174.0 },
{ -323.7, 117.6, -116.0 },
{ -323.7, 117.6, -58.0 },
{ -323.7, 117.6, 0.0 },
{ -323.7, 117.6, 58.0 },
{ -323.7, 117.6, 116.0 },
{ -323.7, 117.6, 174.0 },
{ -323.7, 117.6, 232.0 },
{ -323.7, 117.6, 290.0 },
{ -323.7, 117.6, 348.0 },
{ -264.3, 117.6, -348.0 },
{ -264.3, 117.6, -290.0 },
{ -264.3, 117.6, -232.0 },
{ -264.3, 117.6, -174.0 },
{ -264.3, 117.6, -116.0 },
{ -264.3, 117.6, -58.0 },
{ -264.3, 117.6, 0.0 },
{ -264.3, 117.6, 58.0 },
{ -264.3, 117.6, 116.0 },
{ -264.3, 117.6, 174.0 },
{ -264.3, 117.6, 232.0 },
{ -264.3, 117.6, 290.0 },
{ -264.3, 117.6, 348.0 },
{ -264.3, 176.4, -348.0 },
{ -264.3, 176.4, -290.0 },
{ -264.3, 176.4, -232.0 },
{ -264.3, 176.4, -174.0 },
{ -264.3, 176.4, -116.0 },
{ -264.3, 176.4, -58.0 },
{ -264.3, 176.4, 0.0 },
{ -264.3, 176.4, 58.0 },
{ -264.3, 176.4, 116.0 },
{ -264.3, 176.4, 174.0 },
{ -264.3, 176.4, 232.0 },
{ -264.3, 176.4, 290.0 },
{ -264.3, 176.4, 348.0 },
{ -323.7, 176.4, -348.0 },
{ -323.7, 176.4, -290.0 },
{ -323.7, 176.4, -232.0 },
{ -323.7, 176.4, -174.0 },
{ -323.7, 176.4, -116.0 },
{ -323.7, 176.4, -58.0 },
{ -323.7, 176.4, 0.0 },
{ -323.7, 176.4, 58.0 },
{ -323.7, 176.4, 116.0 },
{ -323.7, 176.4, 174.0 },
{ -323.7, 176.4, 232.0 },
{ -323.7, 176.4, 290.0 },
{ -323.7, 176.4, 348.0 },
{ -29.7, -29.4, -348.0 },
{ -29.7, -29.4, -290.0 },
{ -29.7, -29.4, -232.0 },
{ -29.7, -29.4, -174.0 },
{ -29.7, -29.4, -116.0 },
{ -29.7, -29.4, -58.0 },
{ -29.7, -29.4, 0.0 },
{ -29.7, -29.4, 58.0 },
{ -29.7, -29.4, 116.0 },
{ -29.7, -29.4, 174.0 },
{ -29.7, -29.4, 232.0 },
{ -29.7, -29.4, 290.0 },
{ -29.7, -29.4, 348.0 },
{ 29.7, -29.4, -348.0 },
{ 29.7, -29.4, -290.0 },
{ 29.7, -29.4, -232.0 },
{ 29.7, -29.4, -174.0 },
{ 29.7, -29.4, -116.0 },
{ 29.7, -29.4, -58.0 },
{ 29.7, -29.4, 0.0 },
{ 29.7, -29.4, 58.0 },
{ 29.7, -29.4, 116.0 },
{ 29.7, -29.4, 174.0 },
{ 29.7, -29.4, 232.0 },
{ 29.7, -29.4, 290.0 },
{ 29.7, -29.4, 348.0 },
{ 29.7, 29.4, -348.0 },
{ 29.7, 29.4, -290.0 },
{ 29.7, 29.4, -232.0 },
{ 29.7, 29.4, -174.0 },
{ 29.7, 29.4, -116.0 },
{ 29.7, 29.4, -58.0 },
{ 29.7, 29.4, 0.0 },
{ 29.7, 29.4, 58.0 },
{ 29.7, 29.4, 116.0 },
{ 29.7, 29.4, 174.0 },
{ 29.7, 29.4, 232.0 },
{ 29.7, 29.4, 290.0 },
{ 29.7, 29.4, 348.0 },
{ -29.7, 29.4, -348.0 },
{ -29.7, 29.4, -290.0 },
{ -29.7, 29.4, -232.0 },
{ -29.7, 29.4, -174.0 },
{ -29.7, 29.4, -116.0 },
{ -29.7, 29.4, -58.0 },
{ -29.7, 29.4, 0.0 },
{ -29.7, 29.4, 58.0 },
{ -29.7, 29.4, 116.0 },
{ -29.7, 29.4, 174.0 },
{ -29.7, 29.4, 232.0 },
{ -29.7, 29.4, 290.0 },
{ -29.7, 29.4, 348.0 },
{ -29.7, 117.6, -348.0 },
{ -29.7, 117.6, -290.0 },
{ -29.7, 117.6, -232.0 },
{ -29.7, 117.6, -174.0 },
{ -29.7, 117.6, -116.0 },
{ -29.7, 117.6, -58.0 },
{ -29.7, 117.6, 0.0 },
{ -29.7, 117.6, 58.0 },
{ -29.7, 117.6, 116.0 },
{ -29.7, 117.6, 174.0 },
{ -29.7, 117.6, 232.0 },
{ -29.7, 117.6, 290.0 },
{ -29.7, 117.6, 348.0 },
{ 29.7, 117.6, -348.0 },
{ 29.7, 117.6, -290.0 },
{ 29.7, 117.6, -232.0 },
{ 29.7, 117.6, -174.0 },
{ 29.7, 117.6, -116.0 },
{ 29.7, 117.6, -58.0 },
{ 29.7, 117.6, 0.0 },
{ 29.7, 117.6, 58.0 },
{ 29.7, 117.6, 116.0 },
{ 29.7, 117.6, 174.0 },
{ 29.7, 117.6, 232.0 },
{ 29.7, 117.6, 290.0 },
{ 29.7, 117.6, 348.0 },
{ 29.7, 176.4, -348.0 },
{ 29.7, 176.4, -290.0 },
{ 29.7, 176.4, -232.0 },
{ 29.7, 176.4, -174.0 },
{ 29.7, 176.4, -116.0 },
{ 29.7, 176.4, -58.0 },
{ 29.7, 176.4, 0.0 },
{ 29.7, 176.4, 58.0 },
{ 29.7, 176.4, 116.0 },
{ 29.7, 176.4, 174.0 },
{ 29.7, 176.4, 232.0 },
{ 29.7, 176.4, 290.0 },
{ 29.7, 176.4, 348.0 },
{ -29.7, 176.4, -348.0 },
{ -29.7, 176.4, -290.0 },
{ -29.7, 176.4, -232.0 },
{ -29.7, 176.4, -174.0 },
{ -29.7, 176.4, -116.0 },
{ -29.7, 176.4, -58.0 },
{ -29.7, 176.4, 0.0 },
{ -29.7, 176.4, 58.0 },
{ -29.7, 176.4, 116.0 },
{ -29.7, 176.4, 174.0 },
{ -29.7, 176.4, 232.0 },
{ -29.7, 176.4, 290.0 },
{ -29.7, 176.4, 348.0 },
{ -29.7, 264.6, -348.0 },
{ -29.7, 264.6, -290.0 },
{ -29.7, 264.6, -232.0 },
{ -29.7, 264.6, -174.0 },
{ -29.7, 264.6, -116.0 },
{ -29.7, 264.6, -58.0 },
{ -29.7, 264.6, 0.0 },
{ -29.7, 264.6, 58.0 },
{ -29.7, 264.6, 116.0 },
{ -29.7, 264.6, 174.0 },
{ -29.7, 264.6, 232.0 },
{ -29.7, 264.6, 290.0 },
{ -29.7, 264.6, 348.0 },
{ 29.7, 264.6, -348.0 },
{ 29.7, 264.6, -290.0 },
{ 29.7, 264.6, -232.0 },
{ 29.7, 264.6, -174.0 },
{ 29.7, 264.6, -116.0 },
{ 29.7, 264.6, -58.0 },
{ 29.7, 264.6, 0.0 },
{ 29.7, 264.6, 58.0 },
{ 29.7, 264.6, 116.0 },
{ 29.7, 264.6, 174.0 },
{ 29.7, 264.6, 232.0 },
{ 29.7, 264.6, 290.0 },
{ 29.7, 264.6, 348.0 },
{ 29.7, 323.4, -348.0 },
{ 29.7, 323.4, -290.0 },
{ 29.7, 323.4, -232.0 },
{ 29.7, 323.4, -174.0 },
{ 29.7, 323.4, -116.0 },
{ 29.7, 323.4, -58.0 },
{ 29.7, 323.4, 0.0 },
{ 29.7, 323.4, 58.0 },
{ 29.7, 323.4, 116.0 },
{ 29.7, 323.4, 174.0 },
{ 29.7, 323.4, 232.0 },
{ 29.7, 323.4, 290.0 },
{ 29.7, 323.4, 348.0 },
{ -29.7, 323.4, -348.0 },
{ -29.7, 323.4, -290.0 },
{ -29.7, 323.4, -232.0 },
{ -29.7, 323.4, -174.0 },
{ -29.7, 323.4, -116.0 },
{ -29.7, 323.4, -58.0 },
{ -29.7, 323.4, 0.0 },
{ -29.7, 323.4, 58.0 },
{ -29.7, 323.4, 116.0 },
{ -29.7, 323.4, 174.0 },
{ -29.7, 323.4, 232.0 },
{ -29.7, 323.4, 290.0 },
{ -29.7, 323.4, 348.0 },
{ 117.3, 44.1, -348.0 },
{ 117.3, 44.1, -290.0 },
{ 117.3, 44.1, -232.0 },
{ 117.3, 44.1, -174.0 },
{ 117.3, 44.1, -116.0 },
{ 117.3, 44.1, -58.0 },
{ 117.3, 44.1, 0.0 },
{ 117.3, 44.1, 58.0 },
{ 117.3, 44.1, 116.0 },
{ 117.3, 44.1, 174.0 },
{ 117.3, 44.1, 232.0 },
{ 117.3, 44.1, 290.0 },
{ 117.3, 44.1, 348.0 },
{ 176.7, 44.1, -348.0 },
{ 176.7, 44.1, -290.0 },
{ 176.7, 44.1, -232.0 },
{ 176.7, 44.1, -174.0 },
{ 176.7, 44.1, -116.0 },
{ 176.7, 44.1, -58.0 },
{ 176.7, 44.1, 0.0 },
{ 176.7, 44.1, 58.0 },
{ 176.7, 44.1, 116.0 },
{ 176.7, 44.1, 174.0 },
{ 176.7, 44.1, 232.0 },
{ 176.7, 44.1, 290.0 },
{ 176.7, 44.1, 348.0 },
{ 176.7, 102.9, -348.0 },
{ 176.7, 102.9, -290.0 },
{ 176.7, 102.9, -232.0 },
{ 176.7, 102.9, -174.0 },
{ 176.7, 102.9, -116.0 },
{ 176.7, 102.9, -58.0 },
{ 176.7, 102.9, 0.0 },
{ 176.7, 102.9, 58.0 },
{ 176.7, 102.9, 116.0 },
{ 176.7, 102.9, 174.0 },
{ 176.7, 102.9, 232.0 },
{ 176.7, 102.9, 290.0 },
{ 176.7, 102.9, 348.0 },
{ 117.3, 102.9, -348.0 },
{ 117.3, 102.9, -290.0 },
{ 117.3, 102.9, -232.0 },
{ 117.3, 102.9, -174.0 },
{ 117.3, 102.9, -116.0 },
{ 117.3, 102.9, -58.0 },
{ 117.3, 102.9, 0.0 },
{ 117.3, 102.9, 58.0 },
{ 117.3, 102.9, 116.0 },
{ 117.3, 102.9, 174.0 },
{ 117.3, 102.9, 232.0 },
{ 117.3, 102.9, 290.0 },
{ 117.3, 102.9, 348.0 },
{ 117.3, 191.1, -348.0 },
{ 117.3, 191.1, -290.0 },
{ 117.3, 191.1, -232.0 },
{ 117.3, 191.1, -174.0 },
{ 117.3, 191.1, -116.0 },
{ 117.3, 191.1, -58.0 },
{ 117.3, 191.1, 0.0 },
{ 117.3, 191.1, 58.0 },
{ 117.3, 191.1, 116.0 },
{ 117.3, 191.1, 174.0 },
{ 117.3, 191.1, 232.0 },
{ 117.3, 191.1, 290.0 },
{ 117.3, 191.1, 348.0 },
{ 176.7, 191.1, -348.0 },
{ 176.7, 191.1, -290.0 },
{ 176.7, 191.1, -232.0 },
{ 176.7, 191.1, -174.0 },
{ 176.7, 191.1, -116.0 },
{ 176.7, 191.1, -58.0 },
{ 176.7, 191.1, 0.0 },
{ 176.7, 191.1, 58.0 },
{ 176.7, 191.1, 116.0 },
{ 176.7, 191.1, 174.0 },
{ 176.7, 191.1, 232.0 },
{ 176.7, 191.1, 290.0 },
{ 176.7, 191.1, 348.0 },
{ 176.7, 249.9, -348.0 },
{ 176.7, 249.9, -290.0 },
{ 176.7, 249.9, -232.0 },
{ 176.7, 249.9, -174.0 },
{ 176.7, 249.9, -116.0 },
{ 176.7, 249.9, -58.0 },
{ 176.7, 249.9, 0.0 },
{ 176.7, 249.9, 58.0 },
{ 176.7, 249.9, 116.0 },
{ 176.7, 249.9, 174.0 },
{ 176.7, 249.9, 232.0 },
{ 176.7, 249.9, 290.0 },
{ 176.7, 249.9, 348.0 },
{ 117.3, 249.9, -348.0 },
{ 117.3, 249.9, -290.0 },
{ 117.3, 249.9, -232.0 },
{ 117.3, 249.9, -174.0 },
{ 117.3, 249.9, -116.0 },
{ 117.3, 249.9, -58.0 },
{ 117.3, 249.9, 0.0 },
{ 117.3, 249.9, 58.0 },
{ 117.3, 249.9, 116.0 },
{ 117.3, 249.9, 174.0 },
{ 117.3, 249.9, 232.0 },
{ 117.3, 249.9, 290.0 },
{ 117.3, 249.9, 348.0 },
{ 264.3, -29.4, -348.0 },
{ 264.3, -29.4, -290.0 },
{ 264.3, -29.4, -232.0 },
{ 264.3, -29.4, -174.0 },
{ 264.3, -29.4, -116.0 },
{ 264.3, -29.4, -58.0 },
{ 264.3, -29.4, 0.0 },
{ 264.3, -29.4, 58.0 },
{ 264.3, -29.4, 116.0 },
{ 264.3, -29.4, 174.0 },
{ 264.3, -29.4, 232.0 },
{ 264.3, -29.4, 290.0 },
{ 264.3, -29.4, 348.0 },
{ 323.7, -29.4, -348.0 },
{ 323.7, -29.4, -290.0 },
{ 323.7, -29.4, -232.0 },
{ 323.7, -29.4, -174.0 },
{ 323.7, -29.4, -116.0 },
{ 323.7, -29.4, -58.0 },
{ 323.7, -29.4, 0.0 },
{ 323.7, -29.4, 58.0 },
{ 323.7, -29.4, 116.0 },
{ 323.7, -29.4, 174.0 },
{ 323.7, -29.4, 232.0 },
{ 323.7, -29.4, 290.0 },
{ 323.7, -29.4, 348.0 },
{ 323.7, 29.4, -348.0 },
{ 323.7, 29.4, -290.0 },
{ 323.7, 29.4, -232.0 },
{ 323.7, 29.4, -174.0 },
{ 323.7, 29.4, -116.0 },
{ 323.7, 29.4, -58.0 },
{ 323.7, 29.4, 0.0 },
{ 323.7, 29.4, 58.0 },
{ 323.7, 29.4, 116.0 },
{ 323.7, 29.4, 174.0 },
{ 323.7, 29.4, 232.0 },
{ 323.7, 29.4, 290.0 },
{ 323.7, 29.4, 348.0 },
{ 264.3, 29.4, -348.0 },
{ 264.3, 29.4, -290.0 },
{ 264.3, 29.4, -232.0 },
{ 264.3, 29.4, -174.0 },
{ 264.3, 29.4, -116.0 },
{ 264.3, 29.4, -58.0 },
{ 264.3, 29.4, 0.0 },
{ 264.3, 29.4, 58.0 },
{ 264.3, 29.4, 116.0 },
{ 264.3, 29.4, 174.0 },
{ 264.3, 29.4, 232.0 },
{ 264.3, 29.4, 290.0 },
{ 264.3, 29.4, 348.0 },
{ 264.3, 117.6, -348.0 },
{ 264.3, 117.6, -290.0 },
{ 264.3, 117.6, -232.0 },
{ 264.3, 117.6, -174.0 },
{ 264.3, 117.6, -116.0 },
{ 264.3, 117.6, -58.0 },
{ 264.3, 117.6, 0.0 },
{ 264.3, 117.6, 58.0 },
{ 264.3, 117.6, 116.0 },
{ 264.3, 117.6, 174.0 },
{ 264.3, 117.6, 232.0 },
{ 264.3, 117.6, 290.0 },
{ 264.3, 117.6, 348.0 },
{ 323.7, 117.6, -348.0 },
{ 323.7, 117.6, -290.0 },
{ 323.7, 117.6, -232.0 },
{ 323.7, 117.6, -174.0 },
{ 323.7, 117.6, -116.0 },
{ 323.7, 117.6, -58.0 },
{ 323.7, 117.6, 0.0 },
{ 323.7, 117.6, 58.0 },
{ 323.7, 117.6, 116.0 },
{ 323.7, 117.6, 174.0 },
{ 323.7, 117.6, 232.0 },
{ 323.7, 117.6, 290.0 },
{ 323.7, 117.6, 348.0 },
{ 323.7, 176.4, -348.0 },
{ 323.7, 176.4, -290.0 },
{ 323.7, 176.4, -232.0 },
{ 323.7, 176.4, -174.0 },
{ 323.7, 176.4, -116.0 },
{ 323.7, 176.4, -58.0 },
{ 323.7, 176.4, 0.0 },
{ 323.7, 176.4, 58.0 },
{ 323.7, 176.4, 116.0 },
{ 323.7, 176.4, 174.0 },
{ 323.7, 176.4, 232.0 },
{ 323.7, 176.4, 290.0 },
{ 323.7, 176.4, 348.0 },
{ 264.3, 176.4, -348.0 },
{ 264.3, 176.4, -290.0 },
{ 264.3, 176.4, -232.0 },
{ 264.3, 176.4, -174.0 },
{ 264.3, 176.4, -116.0 },
{ 264.3, 176.4, -58.0 },
{ 264.3, 176.4, 0.0 },
{ 264.3, 176.4, 58.0 },
{ 264.3, 176.4, 116.0 },
{ 264.3, 176.4, 174.0 },
{ 264.3, 176.4, 232.0 },
{ 264.3, 176.4, 290.0 },
{ 264.3, 176.4, 348.0 },
{ 29.7, -264.6, -348.0 },
{ 29.7, -264.6, -290.0 },
{ 29.7, -264.6, -232.0 },
{ 29.7, -264.6, -174.0 },
{ 29.7, -264.6, -116.0 },
{ 29.7, -264.6, -58.0 },
{ 29.7, -264.6, 0.0 },
{ 29.7, -264.6, 58.0 },
{ 29.7, -264.6, 116.0 },
{ 29.7, -264.6, 174.0 },
{ 29.7, -264.6, 232.0 },
{ 29.7, -264.6, 290.0 },
{ 29.7, -264.6, 348.0 },
{ -29.7, -264.6, -348.0 },
{ -29.7, -264.6, -290.0 },
{ -29.7, -264.6, -232.0 },
{ -29.7, -264.6, -174.0 },
{ -29.7, -264.6, -116.0 },
{ -29.7, -264.6, -58.0 },
{ -29.7, -264.6, 0.0 },
{ -29.7, -264.6, 58.0 },
{ -29.7, -264.6, 116.0 },
{ -29.7, -264.6, 174.0 },
{ -29.7, -264.6, 232.0 },
{ -29.7, -264.6, 290.0 },
{ -29.7, -264.6, 348.0 },
{ -29.7, -323.4, -348.0 },
{ -29.7, -323.4, -290.0 },
{ -29.7, -323.4, -232.0 },
{ -29.7, -323.4, -174.0 },
{ -29.7, -323.4, -116.0 },
{ -29.7, -323.4, -58.0 },
{ -29.7, -323.4, 0.0 },
{ -29.7, -323.4, 58.0 },
{ -29.7, -323.4, 116.0 },
{ -29.7, -323.4, 174.0 },
{ -29.7, -323.4, 232.0 },
{ -29.7, -323.4, 290.0 },
{ -29.7, -323.4, 348.0 },
{ 29.7, -323.4, -348.0 },
{ 29.7, -323.4, -290.0 },
{ 29.7, -323.4, -232.0 },
{ 29.7, -323.4, -174.0 },
{ 29.7, -323.4, -116.0 },
{ 29.7, -323.4, -58.0 },
{ 29.7, -323.4, 0.0 },
{ 29.7, -323.4, 58.0 },
{ 29.7, -323.4, 116.0 },
{ 29.7, -323.4, 174.0 },
{ 29.7, -323.4, 232.0 },
{ 29.7, -323.4, 290.0 },
{ 29.7, -323.4, 348.0 },
{ 176.7, -44.1, -348.0 },
{ 176.7, -44.1, -290.0 },
{ 176.7, -44.1, -232.0 },
{ 176.7, -44.1, -174.0 },
{ 176.7, -44.1, -116.0 },
{ 176.7, -44.1, -58.0 },
{ 176.7, -44.1, 0.0 },
{ 176.7, -44.1, 58.0 },
{ 176.7, -44.1, 116.0 },
{ 176.7, -44.1, 174.0 },
{ 176.7, -44.1, 232.0 },
{ 176.7, -44.1, 290.0 },
{ 176.7, -44.1, 348.0 },
{ 117.3, -44.1, -348.0 },
{ 117.3, -44.1, -290.0 },
{ 117.3, -44.1, -232.0 },
{ 117.3, -44.1, -174.0 },
{ 117.3, -44.1, -116.0 },
{ 117.3, -44.1, -58.0 },
{ 117.3, -44.1, 0.0 },
{ 117.3, -44.1, 58.0 },
{ 117.3, -44.1, 116.0 },
{ 117.3, -44.1, 174.0 },
{ 117.3, -44.1, 232.0 },
{ 117.3, -44.1, 290.0 },
{ 117.3, -44.1, 348.0 },
{ 117.3, -102.9, -348.0 },
{ 117.3, -102.9, -290.0 },
{ 117.3, -102.9, -232.0 },
{ 117.3, -102.9, -174.0 },
{ 117.3, -102.9, -116.0 },
{ 117.3, -102.9, -58.0 },
{ 117.3, -102.9, 0.0 },
{ 117.3, -102.9, 58.0 },
{ 117.3, -102.9, 116.0 },
{ 117.3, -102.9, 174.0 },
{ 117.3, -102.9, 232.0 },
{ 117.3, -102.9, 290.0 },
{ 117.3, -102.9, 348.0 },
{ 176.7, -102.9, -348.0 },
{ 176.7, -102.9, -290.0 },
{ 176.7, -102.9, -232.0 },
{ 176.7, -102.9, -174.0 },
{ 176.7, -102.9, -116.0 },
{ 176.7, -102.9, -58.0 },
{ 176.7, -102.9, 0.0 },
{ 176.7, -102.9, 58.0 },
{ 176.7, -102.9, 116.0 },
{ 176.7, -102.9, 174.0 },
{ 176.7, -102.9, 232.0 },
{ 176.7, -102.9, 290.0 },
{ 176.7, -102.9, 348.0 },
{ 176.7, -191.1, -348.0 },
{ 176.7, -191.1, -290.0 },
{ 176.7, -191.1, -232.0 },
{ 176.7, -191.1, -174.0 },
{ 176.7, -191.1, -116.0 },
{ 176.7, -191.1, -58.0 },
{ 176.7, -191.1, 0.0 },
{ 176.7, -191.1, 58.0 },
{ 176.7, -191.1, 116.0 },
{ 176.7, -191.1, 174.0 },
{ 176.7, -191.1, 232.0 },
{ 176.7, -191.1, 290.0 },
{ 176.7, -191.1, 348.0 },
{ 117.3, -191.1, -348.0 },
{ 117.3, -191.1, -290.0 },
{ 117.3, -191.1, -232.0 },
{ 117.3, -191.1, -174.0 },
{ 117.3, -191.1, -116.0 },
{ 117.3, -191.1, -58.0 },
{ 117.3, -191.1, 0.0 },
{ 117.3, -191.1, 58.0 },
{ 117.3, -191.1, 116.0 },
{ 117.3, -191.1, 174.0 },
{ 117.3, -191.1, 232.0 },
{ 117.3, -191.1, 290.0 },
{ 117.3, -191.1, 348.0 },
{ 117.3, -249.9, -348.0 },
{ 117.3, -249.9, -290.0 },
{ 117.3, -249.9, -232.0 },
{ 117.3, -249.9, -174.0 },
{ 117.3, -249.9, -116.0 },
{ 117.3, -249.9, -58.0 },
{ 117.3, -249.9, 0.0 },
{ 117.3, -249.9, 58.0 },
{ 117.3, -249.9, 116.0 },
{ 117.3, -249.9, 174.0 },
{ 117.3, -249.9, 232.0 },
{ 117.3, -249.9, 290.0 },
{ 117.3, -249.9, 348.0 },
{ 176.7, -249.9, -348.0 },
{ 176.7, -249.9, -290.0 },
{ 176.7, -249.9, -232.0 },
{ 176.7, -249.9, -174.0 },
{ 176.7, -249.9, -116.0 },
{ 176.7, -249.9, -58.0 },
{ 176.7, -249.9, 0.0 },
{ 176.7, -249.9, 58.0 },
{ 176.7, -249.9, 116.0 },
{ 176.7, -249.9, 174.0 },
{ 176.7, -249.9, 232.0 },
{ 176.7, -249.9, 290.0 },
{ 176.7, -249.9, 348.0 },
{ 323.7, -117.6, -348.0 },
{ 323.7, -117.6, -290.0 },
{ 323.7, -117.6, -232.0 },
{ 323.7, -117.6, -174.0 },
{ 323.7, -117.6, -116.0 },
{ 323.7, -117.6, -58.0 },
{ 323.7, -117.6, 0.0 },
{ 323.7, -117.6, 58.0 },
{ 323.7, -117.6, 116.0 },
{ 323.7, -117.6, 174.0 },
{ 323.7, -117.6, 232.0 },
{ 323.7, -117.6, 290.0 },
{ 323.7, -117.6, 348.0 },
{ 264.3, -117.6, -348.0 },
{ 264.3, -117.6, -290.0 },
{ 264.3, -117.6, -232.0 },
{ 264.3, -117.6, -174.0 },
{ 264.3, -117.6, -116.0 },
{ 264.3, -117.6, -58.0 },
{ 264.3, -117.6, 0.0 },
{ 264.3, -117.6, 58.0 },
{ 264.3, -117.6, 116.0 },
{ 264.3, -117.6, 174.0 },
{ 264.3, -117.6, 232.0 },
{ 264.3, -117.6, 290.0 },
{ 264.3, -117.6, 348.0 },
{ 264.3, -176.4, -348.0 },
{ 264.3, -176.4, -290.0 },
{ 264.3, -176.4, -232.0 },
{ 264.3, -176.4, -174.0 },
{ 264.3, -176.4, -116.0 },
{ 264.3, -176.4, -58.0 },
{ 264.3, -176.4, 0.0 },
{ 264.3, -176.4, 58.0 },
{ 264.3, -176.4, 116.0 },
{ 264.3, -176.4, 174.0 },
{ 264.3, -176.4, 232.0 },
{ 264.3, -176.4, 290.0 },
{ 264.3, -176.4, 348.0 },
{ 323.7, -176.4, -348.0 },
{ 323.7, -176.4, -290.0 },
{ 323.7, -176.4, -232.0 },
{ 323.7, -176.4, -174.0 },
{ 323.7, -176.4, -116.0 },
{ 323.7, -176.4, -58.0 },
{ 323.7, -176.4, 0.0 },
{ 323.7, -176.4, 58.0 },
{ 323.7, -176.4, 116.0 },
{ 323.7, -176.4, 174.0 },
{ 323.7, -176.4, 232.0 },
{ 323.7, -176.4, 290.0 },
{ 323.7, -176.4, 348.0 },
{ 29.7, -117.6, -348.0 },
{ 29.7, -117.6, -290.0 },
{ 29.7, -117.6, -232.0 },
{ 29.7, -117.6, -174.0 },
{ 29.7, -117.6, -116.0 },
{ 29.7, -117.6, -58.0 },
{ 29.7, -117.6, 0.0 },
{ 29.7, -117.6, 58.0 },
{ 29.7, -117.6, 116.0 },
{ 29.7, -117.6, 174.0 },
{ 29.7, -117.6, 232.0 },
{ 29.7, -117.6, 290.0 },
{ 29.7, -117.6, 348.0 },
{ -29.7, -117.6, -348.0 },
{ -29.7, -117.6, -290.0 },
{ -29.7, -117.6, -232.0 },
{ -29.7, -117.6, -174.0 },
{ -29.7, -117.6, -116.0 },
{ -29.7, -117.6, -58.0 },
{ -29.7, -117.6, 0.0 },
{ -29.7, -117.6, 58.0 },
{ -29.7, -117.6, 116.0 },
{ -29.7, -117.6, 174.0 },
{ -29.7, -117.6, 232.0 },
{ -29.7, -117.6, 290.0 },
{ -29.7, -117.6, 348.0 },
{ -29.7, -176.4, -348.0 },
{ -29.7, -176.4, -290.0 },
{ -29.7, -176.4, -232.0 },
{ -29.7, -176.4, -174.0 },
{ -29.7, -176.4, -116.0 },
{ -29.7, -176.4, -58.0 },
{ -29.7, -176.4, 0.0 },
{ -29.7, -176.4, 58.0 },
{ -29.7, -176.4, 116.0 },
{ -29.7, -176.4, 174.0 },
{ -29.7, -176.4, 232.0 },
{ -29.7, -176.4, 290.0 },
{ -29.7, -176.4, 348.0 },
{ 29.7, -176.4, -348.0 },
{ 29.7, -176.4, -290.0 },
{ 29.7, -176.4, -232.0 },
{ 29.7, -176.4, -174.0 },
{ 29.7, -176.4, -116.0 },
{ 29.7, -176.4, -58.0 },
{ 29.7, -176.4, 0.0 },
{ 29.7, -176.4, 58.0 },
{ 29.7, -176.4, 116.0 },
{ 29.7, -176.4, 174.0 },
{ 29.7, -176.4, 232.0 },
{ 29.7, -176.4, 290.0 },
{ 29.7, -176.4, 348.0 },
{ -117.3, -44.1, -348.0 },
{ -117.3, -44.1, -290.0 },
{ -117.3, -44.1, -232.0 },
{ -117.3, -44.1, -174.0 },
{ -117.3, -44.1, -116.0 },
{ -117.3, -44.1, -58.0 },
{ -117.3, -44.1, 0.0 },
{ -117.3, -44.1, 58.0 },
{ -117.3, -44.1, 116.0 },
{ -117.3, -44.1, 174.0 },
{ -117.3, -44.1, 232.0 },
{ -117.3, -44.1, 290.0 },
{ -117.3, -44.1, 348.0 },
{ -176.7, -44.1, -348.0 },
{ -176.7, -44.1, -290.0 },
{ -176.7, -44.1, -232.0 },
{ -176.7, -44.1, -174.0 },
{ -176.7, -44.1, -116.0 },
{ -176.7, -44.1, -58.0 },
{ -176.7, -44.1, 0.0 },
{ -176.7, -44.1, 58.0 },
{ -176.7, -44.1, 116.0 },
{ -176.7, -44.1, 174.0 },
{ -176.7, -44.1, 232.0 },
{ -176.7, -44.1, 290.0 },
{ -176.7, -44.1, 348.0 },
{ -176.7, -102.9, -348.0 },
{ -176.7, -102.9, -290.0 },
{ -176.7, -102.9, -232.0 },
{ -176.7, -102.9, -174.0 },
{ -176.7, -102.9, -116.0 },
{ -176.7, -102.9, -58.0 },
{ -176.7, -102.9, 0.0 },
{ -176.7, -102.9, 58.0 },
{ -176.7, -102.9, 116.0 },
{ -176.7, -102.9, 174.0 },
{ -176.7, -102.9, 232.0 },
{ -176.7, -102.9, 290.0 },
{ -176.7, -102.9, 348.0 },
{ -117.3, -102.9, -348.0 },
{ -117.3, -102.9, -290.0 },
{ -117.3, -102.9, -232.0 },
{ -117.3, -102.9, -174.0 },
{ -117.3, -102.9, -116.0 },
{ -117.3, -102.9, -58.0 },
{ -117.3, -102.9, 0.0 },
{ -117.3, -102.9, 58.0 },
{ -117.3, -102.9, 116.0 },
{ -117.3, -102.9, 174.0 },
{ -117.3, -102.9, 232.0 },
{ -117.3, -102.9, 290.0 },
{ -117.3, -102.9, 348.0 },
{ -117.3, -191.1, -348.0 },
{ -117.3, -191.1, -290.0 },
{ -117.3, -191.1, -232.0 },
{ -117.3, -191.1, -174.0 },
{ -117.3, -191.1, -116.0 },
{ -117.3, -191.1, -58.0 },
{ -117.3, -191.1, 0.0 },
{ -117.3, -191.1, 58.0 },
{ -117.3, -191.1, 116.0 },
{ -117.3, -191.1, 174.0 },
{ -117.3, -191.1, 232.0 },
{ -117.3, -191.1, 290.0 },
{ -117.3, -191.1, 348.0 },
{ -176.7, -191.1, -348.0 },
{ -176.7, -191.1, -290.0 },
{ -176.7, -191.1, -232.0 },
{ -176.7, -191.1, -174.0 },
{ -176.7, -191.1, -116.0 },
{ -176.7, -191.1, -58.0 },
{ -176.7, -191.1, 0.0 },
{ -176.7, -191.1, 58.0 },
{ -176.7, -191.1, 116.0 },
{ -176.7, -191.1, 174.0 },
{ -176.7, -191.1, 232.0 },
{ -176.7, -191.1, 290.0 },
{ -176.7, -191.1, 348.0 },
{ -176.7, -249.9, -348.0 },
{ -176.7, -249.9, -290.0 },
{ -176.7, -249.9, -232.0 },
{ -176.7, -249.9, -174.0 },
{ -176.7, -249.9, -116.0 },
{ -176.7, -249.9, -58.0 },
{ -176.7, -249.9, 0.0 },
{ -176.7, -249.9, 58.0 },
{ -176.7, -249.9, 116.0 },
{ -176.7, -249.9, 174.0 },
{ -176.7, -249.9, 232.0 },
{ -176.7, -249.9, 290.0 },
{ -176.7, -249.9, 348.0 },
{ -117.3, -249.9, -348.0 },
{ -117.3, -249.9, -290.0 },
{ -117.3, -249.9, -232.0 },
{ -117.3, -249.9, -174.0 },
{ -117.3, -249.9, -116.0 },
{ -117.3, -249.9, -58.0 },
{ -117.3, -249.9, 0.0 },
{ -117.3, -249.9, 58.0 },
{ -117.3, -249.9, 116.0 },
{ -117.3, -249.9, 174.0 },
{ -117.3, -249.9, 232.0 },
{ -117.3, -249.9, 290.0 },
{ -117.3, -249.9, 348.0 },
{ -264.3, -117.6, -348.0 },
{ -264.3, -117.6, -290.0 },
{ -264.3, -117.6, -232.0 },
{ -264.3, -117.6, -174.0 },
{ -264.3, -117.6, -116.0 },
{ -264.3, -117.6, -58.0 },
{ -264.3, -117.6, 0.0 },
{ -264.3, -117.6, 58.0 },
{ -264.3, -117.6, 116.0 },
{ -264.3, -117.6, 174.0 },
{ -264.3, -117.6, 232.0 },
{ -264.3, -117.6, 290.0 },
{ -264.3, -117.6, 348.0 },
{ -323.7, -117.6, -348.0 },
{ -323.7, -117.6, -290.0 },
{ -323.7, -117.6, -232.0 },
{ -323.7, -117.6, -174.0 },
{ -323.7, -117.6, -116.0 },
{ -323.7, -117.6, -58.0 },
{ -323.7, -117.6, 0.0 },
{ -323.7, -117.6, 58.0 },
{ -323.7, -117.6, 116.0 },
{ -323.7, -117.6, 174.0 },
{ -323.7, -117.6, 232.0 },
{ -323.7, -117.6, 290.0 },
{ -323.7, -117.6, 348.0 },
{ -323.7, -176.4, -348.0 },
{ -323.7, -176.4, -290.0 },
{ -323.7, -176.4, -232.0 },
{ -323.7, -176.4, -174.0 },
{ -323.7, -176.4, -116.0 },
{ -323.7, -176.4, -58.0 },
{ -323.7, -176.4, 0.0 },
{ -323.7, -176.4, 58.0 },
{ -323.7, -176.4, 116.0 },
{ -323.7, -176.4, 174.0 },
{ -323.7, -176.4, 232.0 },
{ -323.7, -176.4, 290.0 },
{ -323.7, -176.4, 348.0 },
{ -264.3, -176.4, -348.0 },
{ -264.3, -176.4, -290.0 },
{ -264.3, -176.4, -232.0 },
{ -264.3, -176.4, -174.0 },
{ -264.3, -176.4, -116.0 },
{ -264.3, -176.4, -58.0 },
{ -264.3, -176.4, 0.0 },
{ -264.3, -176.4, 58.0 },
{ -264.3, -176.4, 116.0 },
{ -264.3, -176.4, 174.0 },
{ -264.3, -176.4, 232.0 },
{ -264.3, -176.4, 290.0 },
{ -264.3, -176.4, 348.0 }
};

const double EPSILON = 1e-6;
const double CUBELENGTH = 50;

void print_vector(vector<double> v){
    cout << "(";
    for (int i=0; i < v.size(); i++)
    cout << v[i] << "  ";
    cout << ")" << endl;
}

void print_vector(vector<int> v){
    cout << "(";
    for (int i=0; i < v.size(); i++)
    cout << v[i] << "  ";
    cout << ")" << endl;
}

void print_vector(int v[]){
    cout << "(";
    for (int i=0; i < 30; i++)
    cout << v[i] << "  ";
    cout << ")" << endl;
}

void print_vector(double v[]){
    cout << "(";
    for (int i=0; i < 30; i++)
    cout << v[i] << "  ";
    cout << ")" << endl;
}

// void print_vector(int* v){
//     cout << "(";
//     for (int i=0; i < ; i++)
//     cout << v[i] << "  ";
//     cout << ")" << endl;
// }

//function to calculate dot product of two vectors
double dot_product(std::vector<double> vector_a, std::vector<double> vector_b) {
   double product = 0;
   for (int i = 0; i < 3; i++)
   product += vector_a[i] * vector_b[i];
   return product;
}
//function to calculate cross product of two vectors
std::vector<double> cross_product(std::vector<double> vector_a, std::vector<double> vector_b) {
   std::vector<double> temp (3);

   temp[0] = vector_a[1] * vector_b[2] - vector_a[2] * vector_b[1];
   temp[1] = vector_a[0] * vector_b[2] - vector_a[2] * vector_b[0];
   temp[2] = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0];

   return temp;
}

std::vector<double> vector_sum(std::vector<double> vector_a, std::vector<double> vector_b) {
    std::vector<double> temp (3);

    for (int i = 0; i < 3; i++) temp[i] = vector_a[i] + vector_b[i];
    return temp;
}

std::vector<double> vector_difference(std::vector<double> vector_a, std::vector<double> vector_b) {
    std::vector<double> temp (3);

    for (int i = 0; i < 3; i++) temp[i] = vector_a[i] - vector_b[i];
    return temp;
}


std::vector<double> scalar_product(double s, std::vector<double> v){
    std::vector<double> temp (3);

    //cout << "temp[0] " << temp[0] << endl;
    for (int i = 0; i < 3; i++) temp[i] = s * v[i];

    return temp;
}

double vector_norm(vector<double> v){
    double norm_squared = 0;
    for (int i = 0; i < v.size(); i++) norm_squared += pow(v[i],2);
    return sqrt(norm_squared);
}

// class Collision {
//
//
//     public:
//         vector<int> hit_channels;
//         vector<int> miss_channels;
//         vector<double> track_distances;

std::vector<double> lineplanecollision(std::vector<double> planeNormal,
    std::vector<double> planePoint, std::vector<double> rayDirection, std::vector<double> rayPoint)
{
    double ndotu = dot_product(planeNormal,rayDirection);

    // check if vector parallel to plane
    if (abs(ndotu) < EPSILON) return {};

    double t = - dot_product(planeNormal, vector_difference(rayPoint, planePoint)) / ndotu;
    std::vector<double> collision = vector_sum(rayPoint, scalar_product(t, rayDirection));

    // check if collision point falls outside cube
    for (int i = 0; i < 3; i++){
        if (planeNormal[i] == 0) {
            if (collision[i] > (planePoint[i] + CUBELENGTH / 2.0 + EPSILON) ||
                collision[i] < (planePoint[i] - CUBELENGTH / 2.0 - EPSILON))
            return {};
        }
    }

    return collision;
}


std::vector<vector<double>> linecubecollision(std::vector<double> cubeCenter,
    std::vector<double> rayDirection, std::vector<double> rayPoint){

        std::vector<double> cubeFaces[6] = {
            {cubeCenter[0], cubeCenter[1], cubeCenter[2] + CUBELENGTH / 2.0}, // UP
            {cubeCenter[0], cubeCenter[1] + CUBELENGTH / 2.0, cubeCenter[2]}, // FRONT
            {cubeCenter[0] + CUBELENGTH / 2.0, cubeCenter[1], cubeCenter[2]}, // RIGHT
            {cubeCenter[0], cubeCenter[1], cubeCenter[2] - CUBELENGTH / 2.0}, // DOWN
            {cubeCenter[0], cubeCenter[1] - CUBELENGTH / 2.0, cubeCenter[2]}, // BACK
            {cubeCenter[0] - CUBELENGTH / 2.0, cubeCenter[1], cubeCenter[2]} // LEFT
        };

        std::vector<double> cubeNormals[3] = {
            {0,0,1},
            {0,1,0},
            {1,0,0}
        };

        std::vector<std::vector<double>> cubeCollisions;

        for (int f = 0; f < 6; f++){
            vector<double> facePoint = cubeFaces[f];
            vector<double> faceNormal = cubeNormals[f%3];

            vector<double> collision = lineplanecollision(faceNormal, facePoint, rayDirection, rayPoint);

            if (!collision.empty()) cubeCollisions.push_back(collision);
        }
        return cubeCollisions;
    }

extern "C" void channelcollisions(double line[6], int* hit_channels, int* miss_channels, double* track_distances){
    vector<double> rayPoint = {line[0], line[1], line[2]};
    vector<double> rayDirection = {line[3], line[4], line[5]};

    //std::vector<int> (std::begin(src), std::end(src));

    vector<int> h_channels;
    vector<int> m_channels;
    vector<double> t_distances;

    for (int channel=1; channel <= COORDS.size(); channel++){

        vector<double> cubeCenter = COORDS[channel-1];

        vector<double> CP = vector_difference(cubeCenter, rayPoint);
        double distance_to_line = abs(vector_norm(cross_product(CP, rayDirection)) / vector_norm(rayDirection));

        if (distance_to_line < CUBELENGTH/2.0*sqrt(3) + EPSILON){

            vector<vector<double>> collisions = linecubecollision(cubeCenter, rayDirection, rayPoint);
            if(collisions.size() == 2){
                h_channels.push_back(channel);
                t_distances.push_back(vector_norm(vector_difference(collisions[1], collisions[0])));
            }
            else m_channels.push_back(channel);
        }
    }

    // print_vector(h_channels);
    // print_vector(m_channels);
    // print_vector(t_distances);

    for (int i = 0; i < 30; i++){
        if (i < h_channels.size())
        {
            hit_channels[i] = h_channels[i];
            track_distances[i] = t_distances[i];
        }
        else {
            hit_channels[i] = 0;
            track_distances[i] = 0;
        }

        if (i < m_channels.size()) miss_channels[i] = m_channels[i];
        else miss_channels[i] = 0;

    }


    // for (int i = 0; i < 30; i++) miss_channels[i] = m_channels[i];
    // for (int i = 0; i < 30; i++)
    // track_distances = &t_distances[0];
    // miss_channels = &m_channels[0];
    // hit_channels = &h_channels[0];

    //cout << track_distances[1] << endl;



    //double* a = &v[0];
    //hit_channels = &h_channels[0];
    // std::copy(h_channels.begin(), h_channels.end(), hit_channels);
    // std::copy(m_channels.begin(), m_channels.end(), miss_channels);

    // cout << "here" << endl;
    // std::copy(t_distances.begin(), t_distances.end(), track_distances);
    // cout << "here" << endl;

    // miss_channels = &m_channels[0];
    //track_distances = &t_distances[0];
}



int main(){

    // Using time point and system_clock
    std::chrono::time_point<std::chrono::system_clock> start, end;

    start = std::chrono::system_clock::now();

    int hit_channels[30];
    int miss_channels[30];
    double track_distances[30];
    double line[6] = {0.69888833, 0.35133502, 0.13383552, 0.91514145, 0.07671895,
       0.77288363};
    channelcollisions(line, hit_channels, miss_channels, track_distances);

    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;

    cout << "t= " << elapsed_seconds.count() << endl;

    // cout << *hit_channels <<" " << *(hit_channels + 1) << endl;
    // cout << *(miss_channels) <<" " << *(miss_channels + 1) << endl;
    // cout << *(track_distances) <<" " << *(track_distances + 1) << endl;

    print_vector(hit_channels);
    print_vector(miss_channels);
    print_vector(track_distances);


    return 0;
}