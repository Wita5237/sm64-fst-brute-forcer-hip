#include <unordered_set>
#include <cstring>
#include <string>
#include "FST.hpp"

int main(int argc, char* argv[]) {
    struct FSTOptions o;
    struct FSTData p;

    std::string outFile = "outData.csv";

    bool zMode = false;
    bool quadMode = false;

    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("BitFS Final Speed Transfer Brute Forcer.\n");
            printf("A GPU-based brute forcer to search for working setups for the FST step of the BitFS 0xA TAS.\n\n");
            printf("This program accepts the following options:\n\n");
            printf("    -f <frames>\n");
            printf("         Maximum frames of platform tilt considered.\n");
            printf("             Default: %d\n\n", o.maxFrames);
            printf("    -pu <frames>\n");
            printf("         Number of frames of PU movement for 10k PU route.\n");
            printf("         Currently, only 3 frame routes are supported.\n");
            printf("             Default: %d\n\n", o.nPUFrames);
            printf("    -nx <min_nx> <max_nx> <n_samples>\n");
            printf("         Inclusive range of x normals to be considered, and the number of normals to sample.\n");
            printf("         If min_nx==max_nx then n_samples will be set to 1.\n");
            printf("             Default: %g %g %d\n\n", o.minNX, o.maxNX, o.nSamplesNX);
            printf("    -nxz <min_nxz> <max_nxz> <n_samples>\n");
            printf("         Inclusive range of xz sums to be considered, and the number of z normals to sample.\n");
            printf("         If min_nxz==max_nxz then n_samples will be set to 1.\n");
            printf("         Set these values negative to search negative z normals.\n");
            printf("             Default: %g %g %d\n\n", o.minNXZ, o.maxNXZ, o.nSamplesNXZ);
            printf("    -ny <min_ny> <max_ny> <n_samples>\n");
            printf("         Inclusive range of y normals to be considered, and the number of normals to sample.\n");
            printf("         If min_ny==max_ny then n_samples will be set to 1.\n");
            printf("             Default: %g %g %d\n\n", o.minNY, o.maxNY, o.nSamplesNY);
            printf("    -nz\n");
            printf("         Search by z normal instead of xz sum.\n");
            printf("         Ranges supplied with -nxz will be converted to z normal ranges.\n");
            printf("             Default: off\n\n");
            printf("    -dx <delta_x>\n");
            printf("         x coordinate spacing of positions on the platform.\n");
            printf("             Default: %g\n\n", o.deltaX);
            printf("    -dz <delta_z>\n");
            printf("         z coordinate spacing of positions on the platform.\n");
            printf("             Default: %g\n\n", o.deltaZ);
            printf("    -p <platform_x> <platform_y> <platform_z>\n");
            printf("         Position of the pyramid platform.\n");
            printf("             Default: %g %g %g\n\n", o.platformPos[0], o.platformPos[1], o.platformPos[2]);
            printf("    -q\n");
            printf("         Search all 8 \"quadrants\" simultaneously. Overrides platform position set by -p.\n");
            printf("             Default: off\n\n");
            printf("    -o <path>\n");
            printf("         Path to the output file.\n");
            printf("             Default: %s\n\n", outFile.c_str());
            printf("    -m\n");
            printf("         Minimal output mode. The program will only write a list of normals with solutions to the output file.\n");
            printf("             Default: %d\n\n", o.nThreads);
            printf("    -t <threads>\n");
            printf("         Number of CUDA threads to assign to the program.\n");
            printf("             Default: %d\n\n", o.nThreads);
            printf("    -v\n");
            printf("         Verbose mode. Prints all parameters used in brute force.\n");
            printf("             Default: off\n\n");
            printf("    -h --help\n");
            printf("         Prints this text.\n");
            exit(0);
        }
        else if (!strcmp(argv[i], "-f")) {
            o.maxFrames = std::stoi(argv[i + 1]);

            i += 1;
        }
        else if (!strcmp(argv[i], "-pu")) {
            o.nPUFrames = std::stoi(argv[i + 1]);

            i += 1;
        }
        else if (!strcmp(argv[i], "-t")) {
            o.nThreads = std::stoi(argv[i + 1]);

            i += 1;
        }
        else if (!strcmp(argv[i], "-nx")) {
            o.minNX = std::stof(argv[i + 1]);
            o.maxNX = std::stof(argv[i + 2]);

            if (o.minNX == o.maxNX) {
                o.nSamplesNX = 1;
            }
            else {
                o.nSamplesNX = std::stoi(argv[i + 3]);
            }

            i += 3;
        }
        else if (!strcmp(argv[i], "-nxz")) {
            o.minNXZ = std::stof(argv[i + 1]);
            o.maxNXZ = std::stof(argv[i + 2]);

            if (o.minNXZ == o.maxNXZ) {
                o.nSamplesNXZ = 1;
            }
            else {
                o.nSamplesNXZ = std::stoi(argv[i + 3]);
            }

            i += 3;
        }
        else if (!strcmp(argv[i], "-ny")) {
            o.minNY = std::stof(argv[i + 1]);
            o.maxNY = std::stof(argv[i + 2]);

            if (o.minNY == o.maxNY) {
                o.nSamplesNY = 1;
            }
            else {
                o.nSamplesNY = std::stoi(argv[i + 3]);
            }

            i += 3;
        }
        else if (!strcmp(argv[i], "-nz")) {
            zMode = true;
        }
        else if (!strcmp(argv[i], "-dx")) {
            o.deltaX = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-dz")) {
            o.deltaZ = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-p")) {
            o.platformPos[0] = std::stof(argv[i + 1]);
            o.platformPos[1] = std::stof(argv[i + 2]);
            o.platformPos[2] = std::stof(argv[i + 3]);
            i += 3;
        }
        else if (!strcmp(argv[i], "-q")) {
            quadMode = true;
        }
        else if (!strcmp(argv[i], "-o")) {
            outFile = argv[i + 1];
            i += 1;
        }
        else if (!strcmp(argv[i], "-m")) {
            o.minimalOutput = true;
        }
        else if (!strcmp(argv[i], "-v")) {
            verbose = true;
        }
    }

    if (o.nPUFrames != 3) {
        fprintf(stderr, "Error: This brute forcer currently only supports 3 frame 10k routes. Value selected: %d.", o.nPUFrames);
        return 1;
    }

    int err = initialise_fst_vars(&p);

    if (err != 0) {
        return err;
    }

    o.nSamplesNX = o.minNX == o.maxNX ? 1 : o.nSamplesNX;
    o.nSamplesNY = o.minNY == o.maxNY ? 1 : o.nSamplesNY;
    o.nSamplesNXZ = o.minNXZ == o.maxNXZ ? 1 : o.nSamplesNXZ;

    if (verbose) {
        printf("Max Tilt Frames: %d\n", o.maxFrames);
        printf("Off Platform Frames: %d\n", o.nPUFrames);
        printf("X Normal Range: (%g, %g)\n", o.minNX, o.maxNX);
        if (zMode) {
            printf("Z Normal Range: (%g, %g)\n", o.minNXZ, o.maxNXZ);
        }
        else {
            printf("XZ Sum Range: (%g, %g)\n", o.minNXZ, o.maxNXZ);
        }
        printf("Y Normal Range: (%g, %g)\n", o.minNY, o.maxNY);
        printf("X Normal Samples: %d\n", o.nSamplesNX);
        printf("Z Normal Samples: %d\n", o.nSamplesNXZ);
        printf("Y Normal Samples: %d\n", o.nSamplesNY);
        printf("X Spacing: %g\n", o.deltaX);
        printf("Z Spacing: %g\n", o.deltaZ);
        if (quadMode) {
            printf("Quadrant Search: on\n");
        }
        else {
            printf("Platform Position: (%g, %g, %g)\n", o.platformPos[0], o.platformPos[1], o.platformPos[2]);
            printf("Quadrant Search: off\n");
        }
        printf("\n");
    }

    std::ofstream wf(outFile);

    if (wf.is_open()) {
        wf << std::fixed;
        write_solution_file_header(o.minimalOutput, wf);
    }
    else {
        fprintf(stderr, "Warning: ofstream is not open. No solutions will be written to the output file.\n");
        fprintf(stderr, "         This may be due to an invalid output file path.\n");
    }

    const float deltaNX = (o.nSamplesNX > 1) ? (o.maxNX - o.minNX) / (o.nSamplesNX - 1) : 0;
    const float deltaNY = (o.nSamplesNY > 1) ? (o.maxNY - o.minNY) / (o.nSamplesNY - 1) : 0;
    const float deltaNXZ = (o.nSamplesNXZ > 1) ? (o.maxNXZ - o.minNXZ) / (o.nSamplesNXZ - 1) : 0;
    
    for (int j = 0; j < o.nSamplesNXZ; j++) {
        for (int h = 0; h < o.nSamplesNY; h++) {
            for (int i = 0; i < o.nSamplesNX; i++) {
               for (int quad = 0; quad < (quadMode ? 8 : 1); quad++) {
                    float normX;
                    float normY;
                    float normZ;

                    if (quadMode) {
                        float signX = 2.0 * (quad % 2) - 1.0;
                        float signZ = 2.0 * ((quad / 2) % 2) - 1.0;
                        o.platformPos[0] = (quad / 4) == 0 ? -1945.0f : -2866.0f;
                        o.platformPos[1] = -3225.0f;
                        o.platformPos[2] = -715.0f;

                        normX = signX * fabs(o.minNX + i * deltaNX);
                        normY = o.minNY + h * deltaNY;

                        if (zMode) {
                            normZ = signZ * (fabs(o.minNXZ + j * deltaNXZ));
                        }
                        else {
                            normZ = signZ * (fabs(o.minNXZ + j * deltaNXZ) - fabs(normX));
                        }
                    }
                    else {
                        normX = o.minNX + i * deltaNX;
                        normY = o.minNY + h * deltaNY;

                        if (zMode) {
                            normZ = o.minNXZ + j * deltaNXZ;
                        }
                        else {
                            float normNXZ = o.minNXZ + j * deltaNXZ;
                            float signZ = (normNXZ > 0) - (normNXZ < 0);

                            normZ = signZ * (fabs(normNXZ) - fabs(normX));
                        }
                    }

                    Vec3f testNormal = { normX, normY, normZ };

                    if (check_normal(testNormal, &o, &p, wf)) {
                    }
                }
            }
        }
    }
    
    print_success();

    free_fst_vars(&p);
    wf.close();
}
