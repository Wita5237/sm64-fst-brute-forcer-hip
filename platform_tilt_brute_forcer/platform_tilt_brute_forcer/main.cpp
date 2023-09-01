#include <cstring>
#include <string>
#include "FST.hpp"

int main(int argc, char* argv[]) {
    struct FSTOptions o;
    struct FSTData p;

    std::string outFile = "outData.csv";

    float minNX = 0.1765f;
    float maxNX = 0.1815f;
    float minNY = 0.877f;
    float maxNY = 0.882f;
    float minNXZ = 0.575f;
    float maxNXZ = 0.5775f;
    
    int nSamplesNX = 51;
    int nSamplesNXZ = 26;
    int nSamplesNY = 51;

    bool zMode = false;
    bool quadMode = false;

    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("BitFS Final Speed Transfer Brute Forcer.\n");
            printf("A GPU-based brute forcer to search for working setups for the FST step of the BitFS 0xA TAS.\n\n");
            printf("This program accepts the following options:\n\n");
            printf("  Platform normal search settings:\n");
            printf("    -nx <min_nx> <max_nx> <n_samples>\n");
            printf("         Inclusive range of x normals to be considered, and the number of normals to sample.\n");
            printf("         If min_nx==max_nx then n_samples will be set to 1.\n");
            printf("             Default: %g %g %d\n\n", minNX, maxNX, nSamplesNX);
            printf("    -nxz <min_nxz> <max_nxz> <n_samples>\n");
            printf("         Inclusive range of xz sums to be considered, and the number of z normals to sample.\n");
            printf("         If min_nxz==max_nxz then n_samples will be set to 1.\n");
            printf("         Set these values negative to search negative z normals.\n");
            printf("             Default: %g %g %d\n\n", minNXZ, maxNXZ, nSamplesNXZ);
            printf("    -ny <min_ny> <max_ny> <n_samples>\n");
            printf("         Inclusive range of y normals to be considered, and the number of normals to sample.\n");
            printf("         If min_ny==max_ny then n_samples will be set to 1.\n");
            printf("             Default: %g %g %d\n\n", minNY, maxNY, nSamplesNY);
            printf("    -nz\n");
            printf("         Search by z normal instead of xz sum.\n");
            printf("         Ranges supplied with -nxz will be converted to z normal ranges.\n");
            printf("             Default: off\n\n");
            printf("    -q\n");
            printf("         Search all 8 \"quadrants\" simultaneously. Overrides platform position set by -p.\n");
            printf("             Default: off\n\n");
            printf("  Brute forcer settings:\n");
            printf("    -f <frames>\n");
            printf("         Maximum frames of platform tilt considered.\n");
            printf("             Default: %d\n\n", o.maxFrames);
            printf("    -pu <frames>\n");
            printf("         Number of frames of PU movement for 10k PU route.\n");
            printf("         Currently, only 3 frame routes are supported.\n");
            printf("             Default: %d\n\n", o.nPUFrames);
            printf("    -dx <delta_x>\n");
            printf("         x coordinate spacing of positions on the platform.\n");
            printf("             Default: %g\n\n", o.deltaX);
            printf("    -dz <delta_z>\n");
            printf("         z coordinate spacing of positions on the platform.\n");
            printf("             Default: %g\n\n", o.deltaZ);
            printf("    -p <platform_x> <platform_y> <platform_z>\n");
            printf("         Position of the pyramid platform.\n");
            printf("             Default: %g %g %g\n\n", o.platformPos[0], o.platformPos[1], o.platformPos[2]);
            printf("  Output settings:\n");
            printf("    -o <path>\n");
            printf("         Path to the output file.\n");
            printf("             Default: %s\n\n", outFile.c_str());
            printf("    -m\n");
            printf("         Minimal output mode. The program will only write a list of normals with solutions to the output file.\n");
            printf("             Default: off\n\n");
            printf("  GPU settings:\n");
            printf("    -t <threads>\n");
            printf("         Number of CUDA threads to assign to the program.\n");
            printf("             Default: %d\n\n", o.nThreads);
            printf("    -lsk1 <n_solutions>\n");
            printf("         Maximum number of phase 1 solutions for 10k setup search.\n");
            printf("             Default: %d\n\n", o.limits.MAX_SK_PHASE_ONE);
            printf("    -lsk2a <n_solutions>\n");
            printf("         Maximum number of phase 2a solutions for 10k setup search.\n");
            printf("             Default: %d\n\n", o.limits.MAX_SK_PHASE_TWO_A);
            printf("    -lsk2b <n_solutions>\n");
            printf("         Maximum number of phase 2b solutions for 10k setup search.\n");
            printf("             Default: %d\n\n", o.limits.MAX_SK_PHASE_TWO_B);
            printf("    -lsk2c <n_solutions>\n");
            printf("         Maximum number of phase 2c solutions for 10k setup search.\n");
            printf("             Default: %d\n\n", o.limits.MAX_SK_PHASE_TWO_C);
            printf("    -lsk2d <n_solutions>\n");
            printf("         Maximum number of phase 2d solutions for 10k setup search.\n");
            printf("             Default: %d\n\n", o.limits.MAX_SK_PHASE_TWO_D);
            printf("    -lsk3 <n_solutions>\n");
            printf("         Maximum number of phase 3 solutions for 10k setup search.\n");
            printf("             Default: %d\n\n", o.limits.MAX_SK_PHASE_THREE);
            printf("    -lsk4 <n_solutions>\n");
            printf("         Maximum number of phase 4 solutions for 10k setup search.\n");
            printf("             Default: %d\n\n", o.limits.MAX_SK_PHASE_FOUR);
            printf("    -lsk5 <n_solutions>\n");
            printf("         Maximum number of phase 5 solutions for 10k setup search.\n");
            printf("             Default: %d\n\n", o.limits.MAX_SK_PHASE_FIVE);
            printf("    -lsk6 <n_solutions>\n");
            printf("         Maximum number of phase 6 solutions for 10k setup search.\n");
            printf("             Default: %d\n\n", o.limits.MAX_SK_PHASE_SIX);
            printf("    -lp <n_solutions>\n");
            printf("         Maximum number of platform tilt solutions.\n");
            printf("             Default: %d\n\n", o.limits.MAX_PLAT_SOLUTIONS);
            printf("    -lu <n_solutions>\n");
            printf("         Maximum number of upwarp solutions.\n");
            printf("             Default: %d\n\n", o.limits.MAX_UPWARP_SOLUTIONS);
            printf("    -lsku <n_solutions>\n");
            printf("         Maximum number of slide kick upwarp solutions.\n");
            printf("             Default: %d\n\n", o.limits.MAX_SK_UPWARP_SOLUTIONS);
            printf("    -ls <n_solutions>\n");
            printf("         Maximum number of speed solutions.\n");
            printf("             Default: %d\n\n", o.limits.MAX_SPEED_SOLUTIONS);
            printf("    -l10k <n_solutions>\n");
            printf("         Maximum number of 10k solutions.\n");
            printf("             Default: %d\n\n", o.limits.MAX_10K_SOLUTIONS);
            printf("    -lbd <n_solutions>\n");
            printf("         Maximum number of breakdance solutions.\n");
            printf("             Default: %d\n\n", o.limits.MAX_BD_SOLUTIONS);
            printf("    -ld10k <n_solutions>\n");
            printf("         Maximum number of double 10k solutions.\n");
            printf("             Default: %d\n\n", o.limits.MAX_DOUBLE_10K_SOLUTIONS);
            printf("    -lbp <n_solutions>\n");
            printf("         Maximum number of bully push solutions.\n");
            printf("             Default: %d\n\n", o.limits.MAX_BULLY_PUSH_SOLUTIONS);
            printf("    -lsq <n_squish_spots>\n");
            printf("         Maximum number of squish spots.\n");
            printf("             Default: %d\n\n", o.limits.MAX_SQUISH_SPOTS);
            printf("    -lst <n_strain_setups>\n");
            printf("         Maximum number of strain setups.\n");
            printf("             Default: %d\n\n", o.limits.MAX_STRAIN_SETUPS);
            printf("  Misc settings:\n");
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
            minNX = std::stof(argv[i + 1]);
            maxNX = std::stof(argv[i + 2]);

            if (minNX == maxNX) {
                nSamplesNX = 1;
            }
            else {
                nSamplesNX = std::stoi(argv[i + 3]);
            }

            i += 3;
        }
        else if (!strcmp(argv[i], "-nxz")) {
            minNXZ = std::stof(argv[i + 1]);
            maxNXZ = std::stof(argv[i + 2]);

            if (minNXZ == maxNXZ) {
                nSamplesNXZ = 1;
            }
            else {
                nSamplesNXZ = std::stoi(argv[i + 3]);
            }

            i += 3;
        }
        else if (!strcmp(argv[i], "-ny")) {
            minNY = std::stof(argv[i + 1]);
            maxNY = std::stof(argv[i + 2]);

            if (minNY == maxNY) {
                nSamplesNY = 1;
            }
            else {
                nSamplesNY = std::stoi(argv[i + 3]);
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
        else if (!strcmp(argv[i], "-lsk1")) {
            o.limits.MAX_SK_PHASE_ONE = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-lsk2a")) {
            o.limits.MAX_SK_PHASE_TWO_A = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-lsk2b")) {
            o.limits.MAX_SK_PHASE_TWO_B = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-lsk2c")) {
            o.limits.MAX_SK_PHASE_TWO_C = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-lsk2d")) {
            o.limits.MAX_SK_PHASE_TWO_D = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-lsk3")) {
            o.limits.MAX_SK_PHASE_THREE = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-lsk4")) {
            o.limits.MAX_SK_PHASE_FOUR = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-lsk5")) {
            o.limits.MAX_SK_PHASE_FIVE = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-lsk6")) {
            o.limits.MAX_SK_PHASE_SIX = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-lp")) {
            o.limits.MAX_PLAT_SOLUTIONS = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-lu")) {
            o.limits.MAX_UPWARP_SOLUTIONS = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-lsku")) {
            o.limits.MAX_SK_UPWARP_SOLUTIONS = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-ls")) {
            o.limits.MAX_SPEED_SOLUTIONS = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-l10k")) {
            o.limits.MAX_10K_SOLUTIONS = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-lbd")) {
            o.limits.MAX_BD_SOLUTIONS = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-ld10k")) {
            o.limits.MAX_DOUBLE_10K_SOLUTIONS = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-lbp")) {
            o.limits.MAX_BULLY_PUSH_SOLUTIONS = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-lsq")) {
            o.limits.MAX_SQUISH_SPOTS = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-lst")) {
            o.limits.MAX_STRAIN_SETUPS = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-v")) {
            verbose = true;
        }
    }

    if (o.nPUFrames != 3) {
        fprintf(stderr, "Error: This brute forcer currently only supports 3 frame 10k routes. Value selected: %d.", o.nPUFrames);
        return 1;
    }

    int err = initialise_fst_vars(&p, &o);

    if (err != 0) {
        fprintf(stderr, "       Run this program with --help for details on how to change internal memory limits.\n");
        return err;
    }

    nSamplesNX = minNX == maxNX ? 1 : nSamplesNX;
    nSamplesNY = minNY == maxNY ? 1 : nSamplesNY;
    nSamplesNXZ = minNXZ == maxNXZ ? 1 : nSamplesNXZ;

    if (verbose) {
        printf("Max Tilt Frames: %d\n", o.maxFrames);
        printf("Off Platform Frames: %d\n", o.nPUFrames);
        printf("X Normal Range: (%g, %g)\n", minNX, maxNX);
        if (zMode) {
            printf("Z Normal Range: (%g, %g)\n", minNXZ, maxNXZ);
        }
        else {
            printf("XZ Sum Range: (%g, %g)\n", minNXZ, maxNXZ);
        }
        printf("Y Normal Range: (%g, %g)\n", minNY, maxNY);
        printf("X Normal Samples: %d\n", nSamplesNX);
        printf("Z Normal Samples: %d\n", nSamplesNXZ);
        printf("Y Normal Samples: %d\n", nSamplesNY);
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

    std::ofstream wf;
    initialise_solution_file_stream(wf, outFile, &o);

    const float deltaNX = (nSamplesNX > 1) ? (maxNX - minNX) / (nSamplesNX - 1) : 0;
    const float deltaNY = (nSamplesNY > 1) ? (maxNY - minNY) / (nSamplesNY - 1) : 0;
    const float deltaNXZ = (nSamplesNXZ > 1) ? (maxNXZ - minNXZ) / (nSamplesNXZ - 1) : 0;
    
    for (int j = 0; j < nSamplesNXZ; j++) {
        for (int h = 0; h < nSamplesNY; h++) {
            printf("%d, %d: %.10g, %.10g\n", h, j, minNY + h * deltaNY, minNXZ + j * deltaNXZ);
            for (int i = 0; i < nSamplesNX; i++) {
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

                        normX = signX * fabs(minNX + i * deltaNX);
                        normY = minNY + h * deltaNY;

                        if (zMode) {
                            normZ = signZ * (fabs(minNXZ + j * deltaNXZ));
                        }
                        else {
                            normZ = signZ * (fabs(minNXZ + j * deltaNXZ) - fabs(normX));
                        }
                    }
                    else {
                        normX = minNX + i * deltaNX;
                        normY = minNY + h * deltaNY;

                        if (zMode) {
                            normZ = minNXZ + j * deltaNXZ;
                        }
                        else {
                            float normNXZ = minNXZ + j * deltaNXZ;
                            float signZ = (normNXZ > 0) - (normNXZ < 0);

                            normZ = signZ * (fabs(normNXZ) - fabs(normX));
                        }
                    }

                    float testNormal[3] = {normX, normY, normZ};

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
