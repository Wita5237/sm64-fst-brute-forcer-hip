#include "FST.hpp"
#include "Platform.hpp"
#include "Surface.hpp"
#include "Trig.hpp"

# define M_PI                  3.14159265358979323846  /* pi */

# define MAX_UPWARP_SOLUTIONS       10000000
# define MAX_PLAT_SOLUTIONS           200000

# define MAX_SK_PHASE_ONE              50000
# define MAX_SK_PHASE_TWO_A            50000
# define MAX_SK_PHASE_TWO_B            50000
# define MAX_SK_PHASE_TWO_C          5000000
# define MAX_SK_PHASE_TWO_D          5000000
# define MAX_SK_PHASE_THREE          4000000
# define MAX_SK_PHASE_FOUR           5000000
# define MAX_SK_PHASE_FIVE           5000000
# define MAX_SK_PHASE_SIX             200000

# define MAX_SK_UPWARP_SOLUTIONS      100000
# define MAX_SPEED_SOLUTIONS       100000000
# define MAX_10K_SOLUTIONS            500000
# define MAX_SLIDE_SOLUTIONS         2000000
# define MAX_BD_SOLUTIONS              50000

# define MAX_DOUBLE_10K_SOLUTIONS     300000
# define MAX_BULLY_PUSH_SOLUTIONS     900000

# define MAX_SQUISH_SPOTS               5000
# define MAX_STRAIN_SETUPS            500000

__device__ const int maxF3Turn = 522;
__device__ const int nTenKFloors = 4;
__device__ float tenKFloors[nTenKFloors][9] = {
    { -613.0, -306.0, -4607.0, -4453.0, -3071.0, -2661.0, -0.9361413717, 0.351623833, 0.0 },
    { -613.0, -306.0, -4146.0, -3993.0, -2661.0, -3071.0, 0.936891377, 0.349620432, 0.0 },
    { -7065.0, -6041.0, 307.0, 322.0, -2866.0, -3071.0, 0.0146370763, 0.03655698895, 0.9992243648 },
    { -7065.0, -6553.0, 307.0, 322.0, -2866.0, -3071.0, 0, 0.07297563553, 0.9973337054 }
};

__global__ void initialise_floors() {
    floorsG[0] = SurfaceG(1536, 5478, -262, 922, 5478, -262, 922, 5478, 403);
    floorsG[1] = SurfaceG(1536, 5478, -262, 922, 5478, 403, 1536, 5478, 403);
    floorsG[2] = SurfaceG(2150, 5248, -262, 1536, 5248, -210, 2150, 5248, -210);
    floorsG[3] = SurfaceG(2150, 5248, -262, 1536, 5248, -262, 1536, 5248, -210);
    floorsG[4] = SurfaceG(2765, 5248, -262, 2150, 5248, 403, 2765, 5248, 403);
    floorsG[5] = SurfaceG(2765, 5248, -262, 2150, 5248, -262, 2150, 5248, 403);
    floorsG[6] = SurfaceG(3994, 5018, -262, 2765, 5018, -262, 2765, 5018, -210);
    floorsG[7] = SurfaceG(3994, 5018, -262, 2765, 5018, -210, 3994, 5018, -210);
    floorsG[8] = SurfaceG(7384, 4506, 371, 7513, 4403, -236, 7384, 4506, -181);
    floorsG[9] = SurfaceG(7384, 4506, 371, 7513, 4403, 426, 7513, 4403, -236);
    floorsG[10] = SurfaceG(7384, 4506, -181, 7513, 4403, -236, 7071, 4403, -678);
    floorsG[11] = SurfaceG(7384, 4506, -181, 7071, 4403, -678, 7015, 4506, -549);
    floorsG[12] = SurfaceG(7015, 4506, -549, 7071, 4403, -678, 6407, 4403, -678);
    floorsG[13] = SurfaceG(7015, 4506, -549, 6407, 4403, -678, 6462, 4506, -549);
    floorsG[14] = SurfaceG(6462, 4506, -549, 6407, 4403, -678, 5965, 4403, -236);
    floorsG[15] = SurfaceG(6462, 4506, -549, 5965, 4403, -236, 6094, 4506, -181);
    floorsG[16] = SurfaceG(7384, 4506, -181, 6912, 4019, 170, 7384, 4506, 371);
    floorsG[17] = SurfaceG(7015, 4506, -549, 6912, 4019, 20, 7384, 4506, -181);
    floorsG[18] = SurfaceG(7015, 4506, -549, 6462, 4506, -549, 6661, 4019, -79);
    floorsG[19] = SurfaceG(7384, 4506, -181, 6912, 4019, 20, 6912, 4019, 170);
    floorsG[20] = SurfaceG(6462, 4506, 740, 6561, 4019, 170, 6094, 4506, 371);
    floorsG[21] = SurfaceG(6462, 4506, 740, 6661, 4019, 271, 6561, 4019, 170);
    floorsG[22] = SurfaceG(6094, 4506, 371, 6561, 4019, 170, 6561, 4019, 20);
    floorsG[23] = SurfaceG(6094, 4506, 371, 6561, 4019, 20, 6094, 4506, -181);
    floorsG[24] = SurfaceG(7015, 4506, -549, 6811, 4019, -79, 6912, 4019, 20);
    floorsG[25] = SurfaceG(7015, 4506, -549, 6661, 4019, -79, 6811, 4019, -79);
    floorsG[26] = SurfaceG(6094, 4506, -181, 5965, 4403, -236, 5965, 4403, 426);
    floorsG[27] = SurfaceG(6094, 4506, -181, 5965, 4403, 426, 6094, 4506, 371);
    floorsG[28] = SurfaceG(7071, 4403, 869, 7015, 4506, 740, 6462, 4506, 740);
    floorsG[29] = SurfaceG(7071, 4403, 869, 6462, 4506, 740, 6407, 4403, 869);
    floorsG[30] = SurfaceG(7071, 4403, 869, 7513, 4403, 426, 7384, 4506, 371);
    floorsG[31] = SurfaceG(7071, 4403, 869, 7384, 4506, 371, 7015, 4506, 740);
    floorsG[32] = SurfaceG(5965, 4403, 426, 6407, 4403, 869, 6462, 4506, 740);
    floorsG[33] = SurfaceG(5965, 4403, 426, 6462, 4506, 740, 6094, 4506, 371);
    floorsG[34] = SurfaceG(3994, 4403, -210, 3379, 4403, -210, 3379, 4403, 403);
    floorsG[35] = SurfaceG(4403, 4403, -210, 3994, 4403, -210, 3994, 4403, 403);
    floorsG[36] = SurfaceG(4403, 4403, -210, 3994, 4403, 403, 4403, 4403, 403);
    floorsG[37] = SurfaceG(3994, 4403, -210, 3379, 4403, 403, 3994, 4403, 403);
    floorsG[38] = SurfaceG(1536, 4403, -210, 922, 4403, -210, 922, 4403, 403);
    floorsG[39] = SurfaceG(1536, 4403, -210, 922, 4403, 403, 1536, 4403, 403);
    floorsG[40] = SurfaceG(5555, 4403, 403, 5965, 4403, -210, 5555, 4403, -210);
    floorsG[41] = SurfaceG(1536, 4403, 403, 2150, 4403, 403, 1946, 4403, 198);
    floorsG[42] = SurfaceG(1536, 4403, 403, 1946, 4403, 198, 1741, 4403, 198);
    floorsG[43] = SurfaceG(1536, 4403, 403, 1741, 4403, -6, 1536, 4403, -210);
    floorsG[44] = SurfaceG(1536, 4403, 403, 1741, 4403, 198, 1741, 4403, -6);
    floorsG[45] = SurfaceG(1536, 4403, -210, 1741, 4403, -6, 1946, 4403, -6);
    floorsG[46] = SurfaceG(1536, 4403, -210, 1946, 4403, -6, 2150, 4403, -210);
    floorsG[47] = SurfaceG(2150, 4403, -210, 1946, 4403, 198, 2150, 4403, 403);
    floorsG[48] = SurfaceG(2150, 4403, -210, 1946, 4403, -6, 1946, 4403, 198);
    floorsG[49] = SurfaceG(5555, 4403, 403, 5965, 4403, 403, 5965, 4403, -210);
    floorsG[50] = SurfaceG(5965, 4255, 426, 5965, 4403, 403, 5555, 4403, 403);
    floorsG[51] = SurfaceG(6811, 4019, 271, 7015, 4506, 740, 7384, 4506, 371);
    floorsG[52] = SurfaceG(6661, 4019, 271, 6462, 4506, 740, 7015, 4506, 740);
    floorsG[53] = SurfaceG(6661, 4019, 271, 7015, 4506, 740, 6811, 4019, 271);
    floorsG[54] = SurfaceG(6811, 4019, 271, 7384, 4506, 371, 6912, 4019, 170);
    floorsG[55] = SurfaceG(6561, 4019, 20, 6661, 4019, -79, 6462, 4506, -549);
    floorsG[56] = SurfaceG(6561, 4019, 20, 6462, 4506, -549, 6094, 4506, -181);
    floorsG[57] = SurfaceG(-4965, 3789, 404, -5017, 3789, 404, -5017, 3789, 761);
    floorsG[58] = SurfaceG(-4965, 3789, 404, -5017, 3789, 761, -4965, 3789, 813);
    floorsG[59] = SurfaceG(-5017, 3789, -971, -5017, 3789, -209, -4965, 3789, -209);
    floorsG[60] = SurfaceG(-4965, 3789, -1022, -5017, 3789, -971, -4965, 3789, -209);
    floorsG[61] = SurfaceG(-4965, 3789, 813, -5631, 3789, 761, -5682, 3789, 813);
    floorsG[62] = SurfaceG(-5682, 3789, -1022, -5682, 3789, -971, -5017, 3789, -971);
    floorsG[63] = SurfaceG(-5682, 3789, -1022, -5017, 3789, -971, -4965, 3789, -1022);
    floorsG[64] = SurfaceG(-5682, 3789, -1022, -6911, 3174, -1022, -6911, 3174, -971);
    floorsG[65] = SurfaceG(-5682, 3789, -971, -5682, 3789, -1022, -6911, 3174, -971);
    floorsG[66] = SurfaceG(-5631, 3789, -663, -5682, 3789, -663, -5682, 3789, 813);
    floorsG[67] = SurfaceG(-5631, 3789, -663, -5682, 3789, 813, -5631, 3789, 761);
    floorsG[68] = SurfaceG(-4965, 3789, 813, -5017, 3789, 761, -5631, 3789, 761);
    floorsG[69] = SurfaceG(2765, 3789, -210, 2150, 3789, -210, 2150, 3789, 403);
    floorsG[70] = SurfaceG(2765, 3789, -210, 2150, 3789, 403, 2765, 3789, 403);
    floorsG[71] = SurfaceG(307, 3789, -518, -306, 3789, 710, 307, 3789, 710);
    floorsG[72] = SurfaceG(307, 3789, -518, -306, 3789, -518, -306, 3789, 710);
    floorsG[73] = SurfaceG(-3173, 3686, -876, -3276, 3686, -876, -3276, 3686, -774);
    floorsG[74] = SurfaceG(-3173, 3686, -876, -3276, 3686, -774, -3173, 3686, -774);
    floorsG[75] = SurfaceG(-3276, 3686, -774, -3276, 3686, -876, -3378, 3482, -978);
    floorsG[76] = SurfaceG(-1330, 3686, -872, -1433, 3686, -872, -1433, 3686, -770);
    floorsG[77] = SurfaceG(-1330, 3686, -872, -1433, 3686, -770, -1330, 3686, -770);
    floorsG[78] = SurfaceG(-1433, 3686, -770, -1433, 3686, -872, -1535, 3482, -975);
    floorsG[79] = SurfaceG(-5017, 3686, 404, -4965, 3686, 404, -4965, 3686, -209);
    floorsG[80] = SurfaceG(-5017, 3686, 404, -4965, 3686, -209, -5017, 3686, -209);
    floorsG[81] = SurfaceG(-5631, 3686, -971, -6911, 3072, -971, -6962, 3072, -663);
    floorsG[82] = SurfaceG(-5631, 3686, -971, -6962, 3072, -663, -5631, 3686, -663);
    floorsG[83] = SurfaceG(-4965, 3686, 404, -4484, 3686, 404, -4484, 3686, -209);
    floorsG[84] = SurfaceG(-4965, 3686, 404, -4484, 3686, -209, -4965, 3686, -209);
    floorsG[85] = SurfaceG(-5631, 3686, -971, -5631, 3686, 761, -5017, 3686, 761);
    floorsG[86] = SurfaceG(-5017, 3686, -971, -5631, 3686, -971, -5017, 3686, 761);
    floorsG[87] = SurfaceG(-4607, 3533, 1126, 563, 3533, 1126, 563, 3482, 1024);
    floorsG[88] = SurfaceG(-4607, 3533, 1126, 563, 3482, 1024, -4530, 3482, 1050);
    floorsG[89] = SurfaceG(-4607, 3533, 1126, -4530, 3482, 1050, -4530, 3482, -1106);
    floorsG[90] = SurfaceG(-4607, 3533, 1126, -4530, 3482, -1106, -4607, 3533, -1183);
    floorsG[91] = SurfaceG(5734, 3533, -1183, 563, 3533, -1183, 5658, 3482, -1106);
    floorsG[92] = SurfaceG(563, 3533, -1183, 563, 3482, -1081, 5658, 3482, -1106);
    floorsG[93] = SurfaceG(5734, 3533, -1183, 5658, 3482, -1106, 5734, 3533, 1126);
    floorsG[94] = SurfaceG(-4146, 3533, -209, -4484, 3686, -209, -4484, 3686, 404);
    floorsG[95] = SurfaceG(-4146, 3533, -209, -4484, 3686, 404, -4146, 3533, 404);
    floorsG[96] = SurfaceG(-3378, 3482, -978, -3276, 3686, -876, -3173, 3686, -876);
    floorsG[97] = SurfaceG(-3378, 3482, -978, -3173, 3686, -876, -3071, 3482, -978);
    floorsG[98] = SurfaceG(-3071, 3482, -978, -3173, 3686, -876, -3173, 3686, -774);
    floorsG[99] = SurfaceG(-3327, 3482, -620, -3276, 3686, -774, -3378, 3482, -978);
    floorsG[100] = SurfaceG(-3122, 3482, -620, -3173, 3686, -774, -3276, 3686, -774);
    floorsG[101] = SurfaceG(-3122, 3482, -620, -3276, 3686, -774, -3327, 3482, -620);
    floorsG[102] = SurfaceG(-3071, 3482, -978, -3173, 3686, -774, -3122, 3482, -620);
    floorsG[103] = SurfaceG(-1535, 3482, -975, -1330, 3686, -872, -1228, 3482, -975);
    floorsG[104] = SurfaceG(-1228, 3482, -975, -1330, 3686, -872, -1330, 3686, -770);
    floorsG[105] = SurfaceG(-1228, 3482, -975, -1330, 3686, -770, -1279, 3482, -616);
    floorsG[106] = SurfaceG(-1535, 3482, -975, -1433, 3686, -872, -1330, 3686, -872);
    floorsG[107] = SurfaceG(-1279, 3482, -616, -1330, 3686, -770, -1433, 3686, -770);
    floorsG[108] = SurfaceG(-1279, 3482, -616, -1433, 3686, -770, -1484, 3482, -616);
    floorsG[109] = SurfaceG(-1484, 3482, -616, -1433, 3686, -770, -1535, 3482, -975);
    floorsG[110] = SurfaceG(-4530, 3482, -1106, 563, 3533, -1183, -4607, 3533, -1183);
    floorsG[111] = SurfaceG(-4530, 3482, -1106, 563, 3482, -1081, 563, 3533, -1183);
    floorsG[112] = SurfaceG(5658, 3482, 1050, 563, 3482, 1024, 5734, 3533, 1126);
    floorsG[113] = SurfaceG(563, 3482, 1024, 563, 3533, 1126, 5734, 3533, 1126);
    floorsG[114] = SurfaceG(5658, 3482, -1106, 5658, 3482, 1050, 5734, 3533, 1126);
    floorsG[115] = SurfaceG(-4530, 3482, 1050, -2000, 3482, -26, -4530, 3482, -1106, true);
    floorsG[116] = SurfaceG(-4530, 3482, 1050, 563, 3482, 1024, -2000, 3482, -26, true);
    floorsG[117] = SurfaceG(-4530, 3482, -1106, -2000, 3482, -26, 563, 3482, -1081, true);
    floorsG[118] = SurfaceG(563, 3482, -1081, -2000, 3482, -26, 563, 3482, 1024, true);
    floorsG[119] = SurfaceG(563, 3482, 1024, 3128, 3482, -26, 563, 3482, -1081, true);
    floorsG[120] = SurfaceG(563, 3482, -1081, 3128, 3482, -26, 5658, 3482, -1106, true);
    floorsG[121] = SurfaceG(3128, 3482, -26, 563, 3482, 1024, 5658, 3482, 1050, true);
    floorsG[122] = SurfaceG(5658, 3482, -1106, 3128, 3482, -26, 5658, 3482, 1050, true);
    floorsG[123] = SurfaceG(-7474, 3174, -612, -7525, 2867, 1, -7474, 2867, 1);
    floorsG[124] = SurfaceG(-7474, 3174, -612, -7525, 3174, -612, -7525, 2867, 1);
    floorsG[125] = SurfaceG(-7525, 3174, -612, -7474, 3174, -612, -7321, 3174, -971);
    floorsG[126] = SurfaceG(-7525, 3174, -612, -7321, 3174, -971, -7372, 3174, -1022);
    floorsG[127] = SurfaceG(-7372, 3174, -1021, -7321, 3174, -971, -6911, 3174, -971);
    floorsG[128] = SurfaceG(-7372, 3174, -1021, -6911, 3174, -971, -6911, 3174, -1022);
    floorsG[129] = SurfaceG(-6911, 3174, -663, -6962, 3174, -663, -6962, 3174, -612);
    floorsG[130] = SurfaceG(-6911, 3174, -612, -5682, 3789, -612, -5682, 3789, -663);
    floorsG[131] = SurfaceG(-6911, 3174, -663, -6911, 3174, -612, -5682, 3789, -663);
    floorsG[132] = SurfaceG(-6911, 3174, -663, -6962, 3174, -612, -6911, 3174, -612);
    floorsG[133] = SurfaceG(-6962, 3072, -612, -6962, 3072, -663, -7474, 3072, -612);
    floorsG[134] = SurfaceG(-6962, 3072, -663, -6911, 3072, -971, -7474, 3072, -612);
    floorsG[135] = SurfaceG(-6911, 3072, -971, -7321, 3072, -971, -7474, 3072, -612);
    floorsG[136] = SurfaceG(-6962, 3072, -612, -7474, 3072, -612, -6962, 2765, 52);
    floorsG[137] = SurfaceG(-7474, 3072, -612, -7474, 2765, 1, -6962, 2765, 52);
    floorsG[138] = SurfaceG(6640, 2899, -122, 6640, 2899, 321, 6830, 2899, 321);
    floorsG[139] = SurfaceG(6640, 2899, -122, 6830, 2899, 321, 6957, 2899, 194);
    floorsG[140] = SurfaceG(6640, 2899, -122, 6513, 2899, 194, 6640, 2899, 321);
    floorsG[141] = SurfaceG(6640, 2899, -122, 6957, 2899, 194, 6957, 2899, 4);
    floorsG[142] = SurfaceG(6640, 2899, -122, 6957, 2899, 4, 6830, 2899, -122);
    floorsG[143] = SurfaceG(6640, 2899, -122, 6513, 2899, 4, 6513, 2899, 194);
    floorsG[144] = SurfaceG(-6911, 2867, 1, -6297, 2867, 52, -6297, 2867, 1);
    floorsG[145] = SurfaceG(-6911, 2867, 1, -6911, 2867, 52, -6297, 2867, 52);
    floorsG[146] = SurfaceG(-7474, 2867, 1, -7525, 2867, 1, -7218, 2867, 616);
    floorsG[147] = SurfaceG(-7218, 2867, 616, -7187, 2867, 564, -7474, 2867, 1);
    floorsG[148] = SurfaceG(-7218, 2867, 616, -6297, 2867, 616, -6297, 2867, 564);
    floorsG[149] = SurfaceG(-7218, 2867, 616, -6297, 2867, 564, -7187, 2867, 564);
    floorsG[150] = SurfaceG(-6962, 2867, 1, -6911, 3174, -612, -6962, 3174, -612);
    floorsG[151] = SurfaceG(-6962, 2867, 1, -6911, 2867, 1, -6911, 3174, -612);
    floorsG[152] = SurfaceG(-6962, 2867, 52, -6911, 2867, 1, -6962, 2867, 1);
    floorsG[153] = SurfaceG(-6962, 2867, 52, -6911, 2867, 52, -6911, 2867, 1);
    floorsG[154] = SurfaceG(-6962, 2765, 52, -6297, 2765, 564, -6297, 2765, 52);
    floorsG[155] = SurfaceG(-6962, 2765, 52, -7187, 2765, 564, -6297, 2765, 564);
    floorsG[156] = SurfaceG(-6962, 2765, 52, -7474, 2765, 1, -7187, 2765, 564);
    floorsG[157] = SurfaceG(-2354, 1331, -613, -1637, 1331, 0, -1330, 1331, -306);
    floorsG[158] = SurfaceG(-2354, 1331, -613, -2354, 1331, -306, -2047, 1331, 0);
    floorsG[159] = SurfaceG(-2354, 1331, -613, -2047, 1331, 0, -1637, 1331, 0);
    floorsG[160] = SurfaceG(-2354, 1331, -613, -1330, 1331, -613, -1637, 1331, -921);
    floorsG[161] = SurfaceG(-2354, 1331, -613, -1637, 1331, -921, -2047, 1331, -921);
    floorsG[162] = SurfaceG(-2354, 1331, -613, -1330, 1331, -306, -1330, 1331, -613);
    floorsG[163] = SurfaceG(-1381, 819, -25, -1407, 819, 0, -1356, 819, 0);
    floorsG[164] = SurfaceG(-1356, 819, 614, -1407, 819, 614, -1381, 819, 640);
    floorsG[165] = SurfaceG(-2380, 819, 614, -2431, 819, 614, -2405, 819, 640);
    floorsG[166] = SurfaceG(-2405, 819, -25, -2431, 819, 0, -2380, 819, 0);
    floorsG[167] = SurfaceG(-562, 819, -306, -665, 819, 0, -562, 819, 0);
    floorsG[168] = SurfaceG(-562, 819, -306, -665, 819, -306, -665, 819, 0);
    floorsG[169] = SurfaceG(-665, 819, 614, -1074, 410, 0, -1074, 410, 614);
    floorsG[170] = SurfaceG(-665, 819, 614, -665, 819, 0, -1074, 410, 0);
    floorsG[171] = SurfaceG(-665, 819, 614, -562, 819, 614, -562, 819, 0);
    floorsG[172] = SurfaceG(-562, 819, 0, -562, 819, 614, -50, 410, 614);
    floorsG[173] = SurfaceG(-562, 819, 0, -50, 410, 614, -50, 410, 0);
    floorsG[174] = SurfaceG(-665, 819, 614, -562, 819, 0, -665, 819, 0);
    floorsG[175] = SurfaceG(-767, 819, -613, -1330, 1331, -306, -767, 819, -306);
    floorsG[176] = SurfaceG(-767, 819, -306, -460, 819, -306, -460, 819, -613);
    floorsG[177] = SurfaceG(-767, 819, -306, -460, 819, -613, -767, 819, -613);
    floorsG[178] = SurfaceG(-767, 819, -613, -1330, 1331, -613, -1330, 1331, -306);
    floorsG[179] = SurfaceG(-1330, 788, 614, -2457, 742, 640, -1330, 742, 640);
    floorsG[180] = SurfaceG(-1330, 788, 614, -2457, 788, 614, -2457, 742, 640);
    floorsG[181] = SurfaceG(-1330, 753, 0, -2457, 753, 614, -1330, 753, 614);
    floorsG[182] = SurfaceG(-1330, 753, 0, -2457, 753, 0, -2457, 753, 614);
    floorsG[183] = SurfaceG(-1330, 742, -25, -2457, 742, -25, -2457, 788, 0);
    floorsG[184] = SurfaceG(-1330, 742, -25, -2457, 788, 0, -1330, 788, 0);
    floorsG[185] = SurfaceG(1280, 410, -462, 1178, 410, -462, 1178, 410, -359);
    floorsG[186] = SurfaceG(1280, 410, -462, 1178, 410, -359, 1280, 410, -359);
    floorsG[187] = SurfaceG(1178, 410, -359, 1178, 410, -462, 1075, 205, -564);
    floorsG[188] = SurfaceG(922, 410, 102, 205, 410, 102, 205, 410, 512);
    floorsG[189] = SurfaceG(-5836, 410, 1, -4197, 410, 1, -4505, 410, -305);
    floorsG[190] = SurfaceG(-5836, 410, 1, -4505, 410, -305, -5529, 410, -305);
    floorsG[191] = SurfaceG(-5836, 410, 1, -4197, 410, 617, -4197, 410, 1);
    floorsG[192] = SurfaceG(-5836, 410, 1, -5836, 410, 617, -5529, 410, 924);
    floorsG[193] = SurfaceG(-5836, 410, 1, -5529, 410, 924, -4505, 410, 924);
    floorsG[194] = SurfaceG(-5836, 410, 1, -4505, 410, 924, -4197, 410, 617);
    floorsG[195] = SurfaceG(2458, 410, 102, 1741, 410, 512, 2458, 410, 512);
    floorsG[196] = SurfaceG(922, 410, 102, 205, 410, 512, 922, 410, 512);
    floorsG[197] = SurfaceG(-2969, 410, 617, -2149, 307, 2, -2969, 410, 2);
    floorsG[198] = SurfaceG(-2969, 410, 617, -2149, 307, 617, -2149, 307, 2);
    floorsG[199] = SurfaceG(2458, 410, 102, 1741, 410, 102, 1741, 410, 512);
    floorsG[200] = SurfaceG(2662, 410, 512, 2458, 410, 717, 3277, 410, 717);
    floorsG[201] = SurfaceG(2458, 410, 717, 2662, 410, 512, 2662, 410, 102);
    floorsG[202] = SurfaceG(2458, 410, 717, 2662, 410, 102, 2458, 410, -101);
    floorsG[203] = SurfaceG(2458, 410, -101, 2662, 410, 102, 3072, 410, 102);
    floorsG[204] = SurfaceG(2458, 410, -101, 3072, 410, 102, 3277, 410, -101);
    floorsG[205] = SurfaceG(3277, 410, 717, 3277, 410, -101, 3072, 410, 102);
    floorsG[206] = SurfaceG(3277, 410, 717, 3072, 410, 102, 3072, 410, 512);
    floorsG[207] = SurfaceG(2662, 410, 512, 3277, 410, 717, 3072, 410, 512);
    floorsG[208] = SurfaceG(-50, 410, 614, 205, 410, 0, -50, 410, 0);
    floorsG[209] = SurfaceG(-1433, 410, 614, -1074, 410, 0, -1433, 410, 0);
    floorsG[210] = SurfaceG(-1433, 410, 614, -1074, 410, 614, -1074, 410, 0);
    floorsG[211] = SurfaceG(-50, 410, 614, 205, 410, 614, 205, 410, 0);
    floorsG[212] = SurfaceG(-2354, 256, 1126, 1126, 256, 1126, 1126, 205, 1050);
    floorsG[213] = SurfaceG(-7525, 256, 1126, -2354, 256, 1126, -2354, 205, 1024);
    floorsG[214] = SurfaceG(1126, 256, -1183, -2354, 256, -1183, -2354, 205, -1081);
    floorsG[215] = SurfaceG(1126, 256, -1183, -2354, 205, -1081, 1126, 205, -1106);
    floorsG[216] = SurfaceG(4915, 256, 1126, 4915, 256, -1183, 4838, 205, -1106);
    floorsG[217] = SurfaceG(4915, 256, 1126, 4838, 205, -1106, 4838, 205, 1050);
    floorsG[218] = SurfaceG(-7525, 256, 1126, -2354, 205, 1024, -7449, 205, 1050);
    floorsG[219] = SurfaceG(-7525, 256, 1126, -7449, 205, 1050, -7449, 205, -1106);
    floorsG[220] = SurfaceG(-7525, 256, 1126, -7449, 205, -1106, -7525, 256, -1183);
    floorsG[221] = SurfaceG(2662, 207, 512, -2354, 205, -1081, -2354, 205, 1024, true);
    floorsG[222] = SurfaceG(2662, 207, 512, 2663, 205, 101, -2354, 205, -1081, true);
    floorsG[223] = SurfaceG(1126, 205, 1050, -2354, 205, 1024, -2354, 256, 1126);
    floorsG[224] = SurfaceG(4838, 205, 1050, 1126, 256, 1126, 4915, 256, 1126);
    floorsG[225] = SurfaceG(4838, 205, 1050, 1126, 205, 1050, 1126, 256, 1126);
    floorsG[226] = SurfaceG(1075, 205, -564, 1280, 410, -462, 1382, 205, -564);
    floorsG[227] = SurfaceG(1382, 205, -564, 1280, 410, -462, 1280, 410, -359);
    floorsG[228] = SurfaceG(1075, 205, -564, 1178, 410, -462, 1280, 410, -462);
    floorsG[229] = SurfaceG(1126, 205, -206, 1178, 410, -359, 1075, 205, -564);
    floorsG[230] = SurfaceG(1331, 205, -206, 1280, 410, -359, 1178, 410, -359);
    floorsG[231] = SurfaceG(1331, 205, -206, 1178, 410, -359, 1126, 205, -206);
    floorsG[232] = SurfaceG(1382, 205, -564, 1280, 410, -359, 1331, 205, -206);
    floorsG[233] = SurfaceG(-7449, 205, -1106, -2354, 205, -1081, -2354, 256, -1183);
    floorsG[234] = SurfaceG(1126, 205, -1106, 4838, 205, -1106, 4915, 256, -1183);
    floorsG[235] = SurfaceG(1126, 205, -1106, 4915, 256, -1183, 1126, 256, -1183);
    floorsG[236] = SurfaceG(-7449, 205, -1106, -2354, 256, -1183, -7525, 256, -1183);
    floorsG[237] = SurfaceG(1126, 205, 1050, 3072, 205, 512, 2662, 207, 512, true);
    floorsG[238] = SurfaceG(1126, 205, 1050, 4838, 205, 1050, 3072, 205, 512, true);
    floorsG[239] = SurfaceG(4838, 205, -1106, 3072, 205, 512, 4838, 205, 1050, true);
    floorsG[240] = SurfaceG(4838, 205, -1106, 3072, 205, 102, 3072, 205, 512, true);
    floorsG[241] = SurfaceG(1126, 205, 1050, 2662, 207, 512, -2354, 205, 1024, true);
    floorsG[242] = SurfaceG(2663, 205, 101, 3072, 205, 102, 4838, 205, -1106, true);
    floorsG[243] = SurfaceG(2663, 205, 101, 1126, 205, -1106, -2354, 205, -1081, true);
    floorsG[244] = SurfaceG(2663, 205, 101, 4838, 205, -1106, 1126, 205, -1106, true);
    floorsG[245] = SurfaceG(-7449, 205, -1106, -4919, 205, -26, -2354, 205, -1081, true);
    floorsG[246] = SurfaceG(-2354, 205, -1081, -4919, 205, -26, -2354, 205, 1024, true);
    floorsG[247] = SurfaceG(-7449, 205, 1050, -2354, 205, 1024, -4919, 205, -26, true);
    floorsG[248] = SurfaceG(-7449, 205, 1050, -4919, 205, -26, -7449, 205, -1106, true);
    floorsG[249] = SurfaceG(4403, -665, 819, 4454, -665, 819, 4454, -665, -255);
    floorsG[250] = SurfaceG(4403, -665, 819, 4454, -665, -255, 4403, -665, -204);
    floorsG[251] = SurfaceG(4454, -665, -255, 3379, -665, -204, 4403, -665, -204);
    floorsG[252] = SurfaceG(4454, -665, -255, 3379, -665, -255, 3379, -665, -204);
    floorsG[253] = SurfaceG(3379, -665, -204, 3379, -665, -255, 3072, -665, 102);
    floorsG[254] = SurfaceG(3379, -1279, 819, 3738, -1279, 461, 3738, -1279, 154);
    floorsG[255] = SurfaceG(3738, -1279, 461, 4403, -1279, 819, 4045, -1279, 461);
    floorsG[256] = SurfaceG(3738, -1279, 461, 3379, -1279, 819, 4403, -1279, 819);
    floorsG[257] = SurfaceG(3379, -1279, 819, 3072, -1125, 102, 3072, -1125, 512);
    floorsG[258] = SurfaceG(3379, -1279, 819, 3738, -1279, 154, 3379, -1279, -204);
    floorsG[259] = SurfaceG(3379, -1279, 819, 3379, -1279, -204, 3072, -1125, 102);
    floorsG[260] = SurfaceG(4045, -1279, 154, 4045, -1279, 461, 4403, -1279, 819);
    floorsG[261] = SurfaceG(4045, -1279, 154, 4403, -1279, 819, 4403, -1279, -204);
    floorsG[262] = SurfaceG(3379, -1279, -204, 3738, -1279, 154, 4045, -1279, 154);
    floorsG[263] = SurfaceG(3379, -1279, -204, 4045, -1279, 154, 4403, -1279, -204);
    floorsG[264] = SurfaceG(5070, -2042, 317, 5070, -2042, 215, 4967, -2042, 215);
    floorsG[265] = SurfaceG(5070, -2042, 317, 4967, -2042, 215, 4967, -2042, 317);
    floorsG[266] = SurfaceG(3941, -2042, 317, 3941, -2042, 215, 3839, -2042, 317);
    floorsG[267] = SurfaceG(3941, -2042, 215, 3839, -2042, 215, 3839, -2042, 317);
    floorsG[268] = SurfaceG(4967, -2042, 215, 5070, -2042, 215, 5070, -2124, 164);
    floorsG[269] = SurfaceG(4967, -2042, 215, 5070, -2124, 164, 4967, -2124, 164);
    floorsG[270] = SurfaceG(3941, -2042, 317, 3839, -2042, 317, 3941, -2124, 369);
    floorsG[271] = SurfaceG(3839, -2042, 317, 3839, -2124, 369, 3941, -2124, 369);
    floorsG[272] = SurfaceG(5172, -2093, 215, 5070, -2042, 215, 5070, -2042, 317);
    floorsG[273] = SurfaceG(5172, -2093, 215, 5070, -2042, 317, 5172, -2093, 317);
    floorsG[274] = SurfaceG(3839, -2124, 164, 3839, -2042, 215, 3941, -2042, 215);
    floorsG[275] = SurfaceG(4967, -2124, 369, 5070, -2124, 369, 5070, -2042, 317);
    floorsG[276] = SurfaceG(4967, -2124, 369, 5070, -2042, 317, 4967, -2042, 317);
    floorsG[277] = SurfaceG(3941, -2124, 164, 3839, -2124, 164, 3941, -2042, 215);
    floorsG[278] = SurfaceG(7526, -2149, -40, 6912, -2149, -40, 6912, -2149, 573);
    floorsG[279] = SurfaceG(7526, -2149, -40, 6912, -2149, 573, 7526, -2149, 573);
    floorsG[280] = SurfaceG(1024, -2149, -1289, 1434, -2457, -921, 1434, -2149, -1289);
    floorsG[281] = SurfaceG(1024, -2149, -1289, 1024, -2457, -921, 1434, -2457, -921);
    floorsG[282] = SurfaceG(1690, -2149, -1289, 1382, -2149, -1596, 1075, -2149, -1596);
    floorsG[283] = SurfaceG(1690, -2149, -1289, 1075, -2149, -1596, 768, -2149, -1289);
    floorsG[284] = SurfaceG(1690, -2149, -2211, 1382, -2149, -1596, 1690, -2149, -1289);
    floorsG[285] = SurfaceG(1690, -2149, -2211, 1382, -2149, -1904, 1382, -2149, -1596);
    floorsG[286] = SurfaceG(768, -2149, -1289, 1075, -2149, -1596, 1075, -2149, -1904);
    floorsG[287] = SurfaceG(768, -2149, -1289, 1075, -2149, -1904, 768, -2149, -2211);
    floorsG[288] = SurfaceG(768, -2149, -2211, 1075, -2149, -1904, 1382, -2149, -1904);
    floorsG[289] = SurfaceG(768, -2149, -2211, 1382, -2149, -1904, 1690, -2149, -2211);
    floorsG[290] = SurfaceG(5273, -2196, 215, 5170, -2196, 215, 5170, -2196, 317);
    floorsG[291] = SurfaceG(5273, -2196, 317, 5273, -2196, 215, 5170, -2196, 317);
    floorsG[292] = SurfaceG(6401, -2196, 317, 6299, -2196, 215, 6299, -2196, 317);
    floorsG[293] = SurfaceG(6401, -2196, 317, 6401, -2196, 215, 6299, -2196, 215);
    floorsG[294] = SurfaceG(5273, -2196, 317, 5170, -2196, 317, 5273, -2278, 369);
    floorsG[295] = SurfaceG(5170, -2196, 317, 5170, -2278, 369, 5273, -2278, 369);
    floorsG[296] = SurfaceG(6299, -2196, 215, 6401, -2278, 164, 6299, -2278, 164);
    floorsG[297] = SurfaceG(6299, -2196, 215, 6401, -2196, 215, 6401, -2278, 164);
    floorsG[298] = SurfaceG(6299, -2278, 369, 6401, -2196, 317, 6299, -2196, 317);
    floorsG[299] = SurfaceG(5170, -2278, 164, 5170, -2196, 215, 5273, -2196, 215);
    floorsG[300] = SurfaceG(5273, -2278, 164, 5170, -2278, 164, 5273, -2196, 215);
    floorsG[301] = SurfaceG(6299, -2278, 369, 6401, -2278, 369, 6401, -2196, 317);
    floorsG[302] = SurfaceG(2662, -2457, -306, 2662, -2764, 61, 3072, -2764, 61);
    floorsG[303] = SurfaceG(2662, -2457, -306, 3072, -2764, 61, 3072, -2457, -306);
    floorsG[304] = SurfaceG(2662, -2457, -613, 3072, -2457, -921, 1843, -2457, -921);
    floorsG[305] = SurfaceG(2662, -2457, -613, 2662, -2457, -306, 3072, -2457, -306);
    floorsG[306] = SurfaceG(2662, -2457, -613, 3072, -2457, -306, 3072, -2457, -921);
    floorsG[307] = SurfaceG(2253, -2457, -613, 2662, -2457, -613, 1843, -2457, -921);
    floorsG[308] = SurfaceG(2253, -2457, -613, 1843, -2457, -921, 1843, -2457, -613);
    floorsG[309] = SurfaceG(1843, -2457, -613, 2253, -2457, -306, 2253, -2457, -613);
    floorsG[310] = SurfaceG(1843, -2457, -613, 1434, -2457, -613, 2253, -2457, -306);
    floorsG[311] = SurfaceG(1434, -2457, -613, 1024, -2457, -306, 2253, -2457, -306);
    floorsG[312] = SurfaceG(1434, -2457, -613, 1024, -2457, -921, 1024, -2457, -306);
    floorsG[313] = SurfaceG(1434, -2457, -613, 1434, -2457, -921, 1024, -2457, -921);
    floorsG[314] = SurfaceG(-4453, -2661, -613, -4607, -3071, -306, -4453, -2661, -306);
    floorsG[315] = SurfaceG(-4453, -2661, -613, -4607, -3071, -613, -4607, -3071, -306);
    floorsG[316] = SurfaceG(-4453, -2661, -306, -4146, -2661, -306, -4146, -2661, -613);
    floorsG[317] = SurfaceG(-4453, -2661, -306, -4146, -2661, -613, -4453, -2661, -613);
    floorsG[318] = SurfaceG(-4453, -2743, -306, -4453, -2743, 307, -4146, -2743, 307);
    floorsG[319] = SurfaceG(-4453, -2743, -306, -4146, -2743, 307, -4146, -2743, -306);
    floorsG[320] = SurfaceG(3379, -2764, -347, 4301, -2764, -40, 3994, -2764, -347);
    floorsG[321] = SurfaceG(3379, -2764, -347, 4301, -2764, 573, 4301, -2764, -40);
    floorsG[322] = SurfaceG(3379, -2764, -347, 3072, -2764, -40, 3072, -2764, 573);
    floorsG[323] = SurfaceG(3379, -2764, -347, 3072, -2764, 573, 3379, -2764, 881);
    floorsG[324] = SurfaceG(3379, -2764, -347, 3994, -2764, 881, 4301, -2764, 573);
    floorsG[325] = SurfaceG(3379, -2764, -347, 3379, -2764, 881, 3994, -2764, 881);
    floorsG[326] = SurfaceG(2662, -2764, -347, 2048, -2764, -347, 2048, -2764, 881);
    floorsG[327] = SurfaceG(2662, -2764, -347, 2048, -2764, 881, 2662, -2764, 881);
    floorsG[328] = SurfaceG(3072, -2764, 61, 2662, -2764, 471, 3072, -2764, 471);
    floorsG[329] = SurfaceG(3072, -2764, 61, 2662, -2764, 61, 2662, -2764, 471);
    floorsG[330] = SurfaceG(-7065, -2764, -511, -7986, -2764, 512, -7065, -2764, 512);
    floorsG[331] = SurfaceG(-7065, -2764, -511, -7986, -2764, -511, -7986, -2764, 512);
    floorsG[332] = SurfaceG(-7065, -2764, 307, -6553, -2866, 307, -6553, -2866, -306);
    floorsG[333] = SurfaceG(-7065, -2764, 307, -6553, -2866, -306, -7065, -2764, -306);
    floorsG[334] = SurfaceG(-6553, -2866, 307, -7065, -3071, 322, -6041, -3071, 307);
    floorsG[335] = SurfaceG(-306, -2866, 307, 0, -2866, 922, 0, -2866, -306);
    floorsG[336] = SurfaceG(-306, -2866, 307, 0, -2866, -306, -306, -2866, -306);
    floorsG[337] = SurfaceG(-6041, -2866, -306, -6553, -2866, 307, -6041, -2866, 307);
    floorsG[338] = SurfaceG(-6041, -2866, -306, -6553, -2866, -306, -6553, -2866, 307);
    floorsG[339] = SurfaceG(5222, -2917, 573, 6298, -2917, 573, 6298, -2917, -40);
    floorsG[340] = SurfaceG(5222, -2917, 573, 4301, -2764, -40, 4301, -2764, 573);
    floorsG[341] = SurfaceG(5222, -2917, 573, 5222, -2917, -40, 4301, -2764, -40);
    floorsG[342] = SurfaceG(5222, -2917, 573, 6298, -2917, -40, 5222, -2917, -40);
    floorsG[343] = SurfaceG(-921, -3020, 307, -921, -3020, 922, -306, -2866, 307);
    floorsG[344] = SurfaceG(-921, -3020, 922, 0, -2866, 922, -306, -2866, 307);
    floorsG[345] = SurfaceG(-3993, -3071, -613, -4146, -2661, -306, -3993, -3071, -306);
    floorsG[346] = SurfaceG(-3993, -3071, -613, -4146, -2661, -613, -4146, -2661, -306);
    floorsG[347] = SurfaceG(-7065, -3071, 322, -6553, -2866, 307, -7065, -2866, 307);
    floorsG[348] = SurfaceG(-8191, -3071, 8192, 8192, -3071, -8191, -8191, -3071, -8191, true);
    floorsG[349] = SurfaceG(-8191, -3071, 8192, 8192, -3071, 8192, 8192, -3071, -8191, true);
}

__device__ bool check_inbounds(const float* mario_pos) {
    short x_mod = (short)(int)mario_pos[0];
    short y_mod = (short)(int)mario_pos[1];
    short z_mod = (short)(int)mario_pos[2];

    return (abs(x_mod) < 8192 & abs(y_mod) < 8192 & abs(z_mod) < 8192);
}

__global__ void set_platform_pos(float x, float y, float z) {
    platform_pos[0] = x;
    platform_pos[1] = y;
    platform_pos[2] = z;
}

__global__ void init_reverse_atanG() {
    for (int i = 0; i < 8192; i++) {
        int angle = (65536 + gArctanTableG[i]) % 65536;
        gReverseArctanTableG[angle] = i;
    }
}

__global__ void set_start_triangle(short* tris, float* norms) {
    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 3; y++) {
            startTriangles[x][y][0] = tris[9 * x + 3 * y];
            startTriangles[x][y][1] = tris[9 * x + 3 * y + 1];
            startTriangles[x][y][2] = tris[9 * x + 3 * y + 2];
            startNormals[x][y] = norms[3 * x + y];
            squishTriangles[x][y][0] = tris[18 + 9 * x + 3 * y];
            squishTriangles[x][y][1] = tris[18 + 9 * x + 3 * y + 1];
            squishTriangles[x][y][2] = tris[18 + 9 * x + 3 * y + 2];
            squishNormals[x][y] = norms[6 + 3 * x + y];
        }
    }
}

__global__ void set_platform_normal(float nx, float ny, float nz) {
    platformNormal[0] = nx;
    platformNormal[1] = ny;
    platformNormal[2] = nz;
}

__device__ int16_t atan2_lookupG(float z, float x) {
    int16_t angle = 0;

    if (x == 0) {
        angle = gArctanTableG[0];
    }
    else {
        angle = gArctanTableG[uint16_t(float(float(z / x) * 1024.0 + 0.5))];
    }

    return angle;
}

__device__ int16_t atan2sG(float z, float x) {
    int16_t angle = 0;

    if (x >= 0) {
        if (z >= 0) {
            if (z >= x) {
                angle = atan2_lookupG(x, z);
            }
            else {
                angle = 0x4000 - atan2_lookupG(z, x);
            }
        }
        else {
            z = -z;

            if (z < x) {
                angle = 0x4000 + atan2_lookupG(z, x);
            }
            else {
                angle = 0x8000 - atan2_lookupG(x, z);
            }
        }
    }
    else {
        x = -x;

        if (z < 0) {
            z = -z;

            if (z >= x) {
                angle = 0x8000 + atan2_lookupG(x, z);
            }
            else {
                angle = 0xC000 - atan2_lookupG(z, x);
            }
        }
        else {
            if (z < x) {
                angle = 0xC000 + atan2_lookupG(z, x);
            }
            else {
                angle = -atan2_lookupG(x, z);
            }
        }
    }

    return ((angle + 32768) % 65536) - 32768;
}

__device__ int find_ceil(float* pos, short(&triangles)[4][3][3], float(&normals)[4][3], float* pheight) {
    int idx = -1;

    int16_t x = static_cast<int16_t>(static_cast<int>(pos[0]));
    int16_t y = static_cast<int16_t>(static_cast<int>(pos[1]));
    int16_t z = static_cast<int16_t>(static_cast<int>(pos[2]));

    for (int i = 0; i < 4; i++) {
        int16_t x1 = triangles[i][0][0];
        int16_t z1 = triangles[i][0][2];
        int16_t x2 = triangles[i][1][0];
        int16_t z2 = triangles[i][1][2];

        // Check that the point is within the triangle bounds.
        if ((z1 - z) * (x2 - x1) - (x1 - x) * (z2 - z1) > 0) {
            continue;
        }

        // To slightly save on computation time, set this later.
        int16_t x3 = triangles[i][2][0];
        int16_t z3 = triangles[i][2][2];

        if ((z2 - z) * (x3 - x2) - (x2 - x) * (z3 - z2) > 0) {
            continue;
        }
        if ((z3 - z) * (x1 - x3) - (x3 - x) * (z1 - z3) > 0) {
            continue;
        }

        float nx = normals[i][0];
        float ny = normals[i][1];
        float nz = normals[i][2];
        float oo = -(nx * x1 + ny * triangles[i][0][1] + nz * z1);

        // Find the height of the floor at a given location.
        float height = -(x * nx + nz * z + oo) / ny;
        // Checks for floor interaction with a 78 unit buffer.
        if (y - (height - -78.0f) > 0.0f) {
            continue;
        }

        *pheight = height;
        idx = i;
        break;
    }

    //! (Surface Cucking) Since only the first floor is returned and not the highest,
    //  higher floors can be "cucked" by lower floors.
    return idx;
}

__device__ int find_floor(float* pos, short(&triangles)[2][3][3], float(&normals)[2][3], float* pheight) {
    int idx = -1;

    int16_t x = static_cast<int16_t>(static_cast<int>(pos[0]));
    int16_t y = static_cast<int16_t>(static_cast<int>(pos[1]));
    int16_t z = static_cast<int16_t>(static_cast<int>(pos[2]));

    for (int i = 0; i < 2; i++) {
        int16_t x1 = triangles[i][0][0];
        int16_t z1 = triangles[i][0][2];
        int16_t x2 = triangles[i][1][0];
        int16_t z2 = triangles[i][1][2];

        // Check that the point is within the triangle bounds.
        if ((z1 - z) * (x2 - x1) - (x1 - x) * (z2 - z1) < 0) {
            continue;
        }

        // To slightly save on computation time, set this later.
        int16_t x3 = triangles[i][2][0];
        int16_t z3 = triangles[i][2][2];

        if ((z2 - z) * (x3 - x2) - (x2 - x) * (z3 - z2) < 0) {
            continue;
        }
        if ((z3 - z) * (x1 - x3) - (x3 - x) * (z1 - z3) < 0) {
            continue;
        }

        float nx = normals[i][0];
        float ny = normals[i][1];
        float nz = normals[i][2];
        float oo = -(nx * x1 + ny * triangles[i][0][1] + nz * z1);

        // Find the height of the floor at a given location.
        float height = -(x * nx + nz * z + oo) / ny;
        // Checks for floor interaction with a 78 unit buffer.
        if (y - (height + -78.0f) < 0.0f) {
            continue;
        }

        *pheight = height;
        idx = i;
        break;
    }

    //! (Surface Cucking) Since only the first floor is returned and not the highest,
    //  higher floors can be "cucked" by lower floors.
    return idx;
}

__device__ int find_floor(float* position, SurfaceG** floor, float& floor_y, SurfaceG floor_set[], int n_floor_set) {
    short x = (short)(int)position[0];
    short y = (short)(int)position[1];
    short z = (short)(int)position[2];

    int floor_idx = -1;

    for (int i = 0; i < n_floor_set; ++i) {
        if (x < floor_set[i].min_x || x > floor_set[i].max_x || z < floor_set[i].min_z || z > floor_set[i].max_z) {
            continue;
        }

        if ((floor_set[i].vertices[0][2] - z) * (floor_set[i].vertices[1][0] - floor_set[i].vertices[0][0]) - (floor_set[i].vertices[0][0] - x) * (floor_set[i].vertices[1][2] - floor_set[i].vertices[0][2]) < 0) {
            continue;
        }
        if ((floor_set[i].vertices[1][2] - z) * (floor_set[i].vertices[2][0] - floor_set[i].vertices[1][0]) - (floor_set[i].vertices[1][0] - x) * (floor_set[i].vertices[2][2] - floor_set[i].vertices[1][2]) < 0) {
            continue;
        }
        if ((floor_set[i].vertices[2][2] - z) * (floor_set[i].vertices[0][0] - floor_set[i].vertices[2][0]) - (floor_set[i].vertices[2][0] - x) * (floor_set[i].vertices[0][2] - floor_set[i].vertices[2][2]) < 0) {
            continue;
        }

        float height = -(x * floor_set[i].normal[0] + floor_set[i].normal[2] * z + floor_set[i].origin_offset) / floor_set[i].normal[1];

        if (y - (height + -78.0f) < 0.0f) {
            continue;
        }

        floor_y = height;
        *floor = &floor_set[i];
        floor_idx = i;
        break;
    }

    return floor_idx;
}

__device__ float find_closest_mag(float target) {
    int minIdx = -1;
    int maxIdx = magCount;

    while (maxIdx > minIdx + 1) {
        int midIdx = (maxIdx + minIdx) / 2;

        if (target < magSet[midIdx]) {
            maxIdx = midIdx;
        }
        else {
            minIdx = midIdx;
        }
    }

    if (minIdx == -1) {
        return magSet[maxIdx];
    }
    else if (maxIdx == magCount) {
        return magSet[minIdx];
    }
    else if (target - magSet[minIdx] < magSet[maxIdx] - target) {
        return magSet[minIdx];
    }
    else {
        return magSet[maxIdx];
    }
}

__global__ void init_camera_angles() {
    for (int i = 0; i < 65536; i += 16) {
        int angle = atan2sG(gCosineTableG[i >> 4], gSineTableG[i >> 4]);
        angle = (65536 + angle) % 65536;

        validCameraAngle[angle] = true;
    }
}

__global__ void init_mag_set() {
    bool magCheck[4097];

    for (int i = 0; i <= 4096; i++) {
        magCheck[i] = false;
    }

    for (int x = -128; x < 128; x++) {
        for (int y = -128; y < 128; y++) {
            int xS;
            if (x < 8) {
                if (x > -8) {
                    xS = 0;
                }
                else {
                    xS = x + 6;
                }
            }
            else {
                xS = x - 6;
            }
            int yS;
            if (y < 8) {
                if (y > -8) {
                    yS = 0;
                }
                else {
                    yS = y + 6;
                }
            }
            else {
                yS = y - 6;
            }

            int mag2 = xS * xS + yS * yS;
            mag2 = mag2 > 4096 ? 4096 : mag2;

            magCheck[mag2] = true;
        }
    }

    for (int i = 0; i <= 4096; i++) {
        if (magCheck[i]) {
            float mag = sqrtf((float)i);
            mag = (mag / 64.0f) * (mag / 64.0f) * 32.0f;
            magSet[magCount] = mag;
            magCount++;
        }
    }
}

__device__ int atan2b(double z, double x) {
    double A = 65536 * atan2(x, z) / (2 * M_PI);
    A = fmod(65536.0 + A, 65536.0);
    int lower = 0;
    int upper = 8192;

    while (upper - lower > 1) {
        int mid = (upper + lower) / 2;

        if (fmod(65536.0 + gArctanTableG[mid], 65536.0) > A) {
            upper = mid;
        }
        else {
            lower = mid;
        }
    }

    return lower;
}

__device__ float find_pre10K_speed(float post10KSpeed, float strainX, float strainZ, float& post10KVelX, float& post10KVelZ, int solIdx) {
    struct SKPhase6* sol6 = &(sk6Solutions[solIdx]);
    struct SKPhase5* sol5 = &(sk5Solutions[sol6->p5Idx]);
    struct SKPhase4* sol4 = &(sk4Solutions[sol5->p4Idx]);
    struct SKPhase3* sol3 = &(sk3Solutions[sol4->p3Idx]);
    struct SKPhase2* sol2 = (sol3->p2Type / 2 == 0) ? ((sol3->p2Type % 2 == 0) ? &(sk2ASolutions[sol3->p2Idx]) : &(sk2BSolutions[sol3->p2Idx])) : ((sol3->p2Type % 2 == 0) ? &(sk2CSolutions[sol3->p2Idx]) : &(sk2DSolutions[sol3->p2Idx]));

    float pre10KSpeed = NAN;
    post10KVelX = NAN;
    post10KVelZ = NAN;

    float mag = sqrtf((float)(sol5->stickX * sol5->stickX + sol5->stickY * sol5->stickY));

    float xS = sol5->stickX;
    float yS = sol5->stickY;

    if (mag > 64.0f) {
        xS = xS * (64.0f / mag);
        yS = yS * (64.0f / mag);
        mag = 64.0f;
    }

    float intendedMag = ((mag / 64.0f) * (mag / 64.0f)) * 32.0f;
    int intendedYaw = atan2sG(-yS, xS) + sol4->cameraYaw;
    int intendedDYaw = intendedYaw - sol5->f1Angle;
    intendedDYaw = (65536 + (intendedDYaw % 65536)) % 65536;

    double w = intendedMag * gCosineTableG[intendedDYaw >> 4];
    double eqB = (50.0 + 147200.0 / w);
    double eqC = -(320000.0 / w) * post10KSpeed;
    double eqDet = eqB * eqB - eqC;

    if (eqDet >= 0) {
        pre10KSpeed = sqrt(eqDet) - eqB;

        if (pre10KSpeed >= 0) {
            bool searchLoop = true;

            float upperSpeed = 2.0f * pre10KSpeed;
            float lowerSpeed = 0.0f;

            while (searchLoop) {
                pre10KSpeed = fmaxf((upperSpeed + lowerSpeed) / 2.0f, nextafterf(lowerSpeed, INFINITY));

                float pre10KVelX = (pre10KSpeed * gSineTableG[sol2->f2Angle >> 4]) + strainX;
                float pre10KVelZ = (pre10KSpeed * gCosineTableG[sol2->f2Angle >> 4]) + strainZ;

                post10KVelX = pre10KVelX;
                post10KVelZ = pre10KVelZ;

                float oldSpeed = sqrtf(post10KVelX * post10KVelX + post10KVelZ * post10KVelZ);

                post10KVelX += post10KVelZ * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;
                post10KVelZ -= post10KVelX * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;

                float newSpeed = sqrtf(post10KVelX * post10KVelX + post10KVelZ * post10KVelZ);

                post10KVelX = post10KVelX * oldSpeed / newSpeed;
                post10KVelZ = post10KVelZ * oldSpeed / newSpeed;

                post10KVelX += 7.0f * tenKFloors[sol2->tenKFloorIdx][6];
                post10KVelZ += 7.0f * tenKFloors[sol2->tenKFloorIdx][8];

                float forward = gCosineTableG[intendedDYaw >> 4] * (0.5f + 0.5f * pre10KSpeed / 100.0f);
                float lossFactor = intendedMag / 32.0f * forward * 0.02f + 0.92f;

                post10KVelX *= lossFactor;
                post10KVelZ *= lossFactor;

                float post10KSpeedTest = -sqrtf(post10KVelX * post10KVelX + post10KVelZ * post10KVelZ);

                if (post10KSpeedTest == post10KSpeed) {
                    searchLoop = false;
                }
                else {
                    if (post10KSpeedTest < post10KSpeed) {
                        upperSpeed = pre10KSpeed;
                    }
                    else {
                        lowerSpeed = pre10KSpeed;
                    }

                    if (nextafterf(lowerSpeed, INFINITY) == upperSpeed) {
                        searchLoop = false;
                        pre10KSpeed = NAN;
                        post10KVelX = NAN;
                        post10KVelZ = NAN;
                    }
                }
            }
        }
    }

    return pre10KSpeed;
}

__device__ void adjust_position_to_ints(float* a, float* b, float p[2][3]) {
    short x1 = (short)(int)a[0];
    short z1 = (short)(int)a[2];

    int roundDirX = (x1 > a[0]) - (x1 < a[0]);
    int roundDirZ = (z1 > a[2]) - (z1 < a[2]);

    float z2 = ((double)p[1][2] - (double)p[0][2]) * ((double)x1 - (double)p[0][0]) / ((double)p[1][0] - (double)p[0][0]) + (double)p[0][2];
    float x2 = ((double)p[1][0] - (double)p[0][0]) * ((double)z1 - (double)p[0][2]) / ((double)p[1][2] - (double)p[0][2]) + (double)p[0][0];

    int lookDirX = (x2 > x1) - (x2 < x1);
    int lookDirZ = (z2 > z1) - (z2 < z1);

    x2 = (short)(int)x2 - roundDirX;
    z2 = (short)(int)z2 - roundDirZ;

    x2 = (lookDirX == roundDirX) ? nextafterf(x2, roundDirX * INFINITY) : x2;
    z2 = (lookDirZ == roundDirZ) ? nextafterf(z2, roundDirZ * INFINITY) : z2;

    double rX = ((double)x2 - (double)a[0]) / ((double)b[0] - (double)a[0]);
    double rZ = ((double)z2 - (double)a[2]) / ((double)b[2] - (double)a[2]);

    if (fabs(rX) < fabs(rZ)) {
        a[0] = (b[0] - a[0]) * rX + a[0];
        a[1] = (b[1] - a[1]) * rX + a[1];
        a[2] = (b[2] - a[2]) * rX + a[2];
    }
    else {
        a[0] = (b[0] - a[0]) * rZ + a[0];
        a[1] = (b[1] - a[1]) * rZ + a[1];
        a[2] = (b[2] - a[2]) * rZ + a[2];
    }
}

__global__ void test_speed_solution(int* squishEdges, const int nPoints, float floorNormalY, int uphillAngle, float maxSlidingSpeed, float maxSlidingSpeedToPlatform) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSpeedSolutions, MAX_SPEED_SOLUTIONS)) {
        struct SpeedSolution* sol = &(speedSolutions[idx]);
        struct SKUpwarpSolution* skuwSol = &(skuwSolutions[sol->skuwSolutionIdx]);
        struct UpwarpSolution* uwSol = &(upwarpSolutions[skuwSol->uwIdx]);
        struct PlatformSolution* platSol = &(platSolutions[uwSol->platformSolutionIdx]);
        struct SKPhase6* sol6 = &(sk6Solutions[skuwSol->skIdx]);
        struct SKPhase5* sol5 = &(sk5Solutions[sol6->p5Idx]);
        struct SKPhase4* sol4 = &(sk4Solutions[sol5->p4Idx]);
        struct SKPhase3* sol3 = &(sk3Solutions[sol4->p3Idx]);
        struct SKPhase2* sol2 = (sol3->p2Type / 2 == 0) ? ((sol3->p2Type % 2 == 0) ? &(sk2ASolutions[sol3->p2Idx]) : &(sk2BSolutions[sol3->p2Idx])) : ((sol3->p2Type % 2 == 0) ? &(sk2CSolutions[sol3->p2Idx]) : &(sk2DSolutions[sol3->p2Idx]));
        struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);

        float returnVelX;
        float returnVelZ;
        float pre10KSpeed = find_pre10K_speed(sol->returnSpeed, sol->xStrain, sol->zStrain, returnVelX, returnVelZ, skuwSol->skIdx);

        if (!isnan(pre10KSpeed)) {
            float frame2Position[3] = { platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (returnVelX / 4.0f), platSol->returnPosition[1], platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (returnVelZ / 4.0f) };

            SurfaceG* floor;
            float floorHeight;

            int floorIdx = find_floor(frame2Position, &floor, floorHeight, floorsG, total_floorsG);

            if (floorIdx != -1 && floor->normal[1] == tenKFloors[sol2->tenKFloorIdx][7] && floorHeight < platSol->returnPosition[1] && floorHeight >= platSol->returnPosition[1] - 78.0f && floorHeight > -2971.0f) {
                int returnSlideYaw = atan2sG(returnVelZ, returnVelX);
                int newFacingDYaw = (short)(sol2->f2Angle - returnSlideYaw);

                if (newFacingDYaw > 0 && newFacingDYaw <= 0x4000) {
                    if ((newFacingDYaw -= 0x200) < 0) {
                        newFacingDYaw = 0;
                    }
                }
                else if (newFacingDYaw > -0x4000 && newFacingDYaw < 0) {
                    if ((newFacingDYaw += 0x200) > 0) {
                        newFacingDYaw = 0;
                    }
                }
                else if (newFacingDYaw > 0x4000 && newFacingDYaw < 0x8000) {
                    if ((newFacingDYaw += 0x200) > 0x8000) {
                        newFacingDYaw = 0x8000;
                    }
                }
                else if (newFacingDYaw > -0x8000 && newFacingDYaw < -0x4000) {
                    if ((newFacingDYaw -= 0x200) < -0x8000) {
                        newFacingDYaw = -0x8000;
                    }
                }

                int returnFaceAngle = returnSlideYaw + newFacingDYaw;
                returnFaceAngle = (65536 + returnFaceAngle) % 65536;

                float postReturnVelX = sol->returnSpeed * gSineTableG[returnFaceAngle >> 4];
                float postReturnVelZ = sol->returnSpeed * gCosineTableG[returnFaceAngle >> 4];

                float intendedPosition[3] = { platSol->returnPosition[0] + postReturnVelX / 4.0f, platSol->returnPosition[1], platSol->returnPosition[2] + postReturnVelZ / 4.0f };

                bool outOfBoundsTest = !check_inbounds(intendedPosition);

                for (int f = 0; outOfBoundsTest && f < 3; f++) {
                    intendedPosition[0] = platSol->landingPositions[f][0] + platSol->landingFloorNormalsY[f] * (postReturnVelX / 4.0f);
                    intendedPosition[1] = platSol->landingPositions[f][1];
                    intendedPosition[2] = platSol->landingPositions[f][2] + platSol->landingFloorNormalsY[f] * (postReturnVelZ / 4.0f);

                    outOfBoundsTest = !check_inbounds(intendedPosition);
                }

                if (outOfBoundsTest) {
                    frame2Position[1] = floorHeight;

                    float pre10KVelX = pre10KSpeed * gSineTableG[sol2->f2Angle >> 4] + sol->xStrain;
                    float pre10KVelZ = pre10KSpeed * gCosineTableG[sol2->f2Angle >> 4] + sol->zStrain;

                    float frame1Position[3] = { frame2Position[0], frame2Position[1], frame2Position[2] };

                    bool inBoundsTest = true;

                    for (int q = 0; q < sol1->q2; q++) {
                        frame1Position[0] = frame1Position[0] - (pre10KVelX / 4.0f);
                        frame1Position[2] = frame1Position[2] - (pre10KVelZ / 4.0f);

                        if (!check_inbounds(frame1Position)) {
                            inBoundsTest = false;
                            break;
                        }
                    }

                    if (inBoundsTest) {
                        floorIdx = find_floor(frame1Position, startTriangles, startNormals, &floorHeight);

                        if (floorIdx != -1 && startNormals[floorIdx][1] == floorNormalY && floorHeight + (sol1->q2 * 20.0f / 4.0f) < frame2Position[1] && floorHeight + (sol1->q2 * 20.0f / 4.0f) >= frame2Position[1] - 78.0f && floorHeight > -3071.0f) {
                            frame1Position[1] = floorHeight;

                            float startSpeed = pre10KSpeed + 1.0f;
                            startSpeed = startSpeed - sol->forwardStrain;
                            startSpeed = startSpeed + 0.35f;

                            float startPositions[2][2][3];
                            int intersectionIdxs[2];
                            int intersections = 0;

                            for (int i = 0; i < nPoints; i++) {
                                if (squishEdges[i] != -1 && nSquishSpots[squishEdges[i]] > 0) {
                                    int surfAngle = atan2sG(squishCeilingNormals[squishEdges[i]][2], squishCeilingNormals[squishEdges[i]][0]);
                                    surfAngle = (65536 + surfAngle) % 65536;

                                    float xPushVel = gSineTableG[surfAngle >> 4] * 10.0f;
                                    float zPushVel = gCosineTableG[surfAngle >> 4] * 10.0f;

                                    int squishFloorIdx = (squishEdges[i] == 0 || squishEdges[i] == 2) ? 0 : 1;

                                    float xOffset = squishNormals[squishFloorIdx][1] * (xPushVel / 4.0f);
                                    float zOffset = squishNormals[squishFloorIdx][1] * (zPushVel / 4.0f);

                                    float p[2][3];
                                    float q[2][3];
                                    int pIdx = 0;

                                    for (int j = 0; j < 3; j++) {
                                        if (startCeilingTriangles[squishEdges[i]][j][0] != platform_pos[0] || startCeilingTriangles[squishEdges[i]][j][1] != platform_pos[1] || startCeilingTriangles[squishEdges[i]][j][2] != platform_pos[2]) {
                                            float oo = -(startNormals[squishFloorIdx][0] * startTriangles[squishFloorIdx][0][0] + startNormals[squishFloorIdx][1] * startTriangles[squishFloorIdx][0][1] + startNormals[squishFloorIdx][2] * startTriangles[squishFloorIdx][0][2]);

                                            p[pIdx][0] = squishCeilingTriangles[squishEdges[i]][j][0] + xOffset;
                                            p[pIdx][2] = squishCeilingTriangles[squishEdges[i]][j][2] + zOffset;
                                            p[pIdx][1] = -(p[pIdx][0] * startNormals[squishFloorIdx][0] + startNormals[squishFloorIdx][2] * p[pIdx][2] + oo) / startNormals[squishFloorIdx][1];

                                            q[pIdx][0] = squishCeilingTriangles[squishEdges[i]][j][0];
                                            q[pIdx][2] = squishCeilingTriangles[squishEdges[i]][j][2];
                                            q[pIdx][1] = -(q[pIdx][0] * startNormals[squishFloorIdx][0] + startNormals[squishFloorIdx][2] * q[pIdx][2] + oo) / startNormals[squishFloorIdx][1];

                                            pIdx++;
                                        }
                                    }

                                    double eqA1 = ((double)p[1][0] - (double)p[0][0]) * ((double)p[1][0] - (double)p[0][0]) + ((double)p[1][2] - (double)p[0][2]) * ((double)p[1][2] - (double)p[0][2]);
                                    double eqB1 = 2.0 * (((double)p[1][0] - (double)p[0][0]) * ((double)p[0][0] - frame1Position[0]) + ((double)p[1][2] - (double)p[0][2]) * ((double)p[0][2] - frame1Position[2]));
                                    double eqC1 = ((double)p[0][0] - frame1Position[0]) * ((double)p[0][0] - frame1Position[0]) + ((double)p[0][2] - frame1Position[2]) * ((double)p[0][2] - frame1Position[2]) - ((double)startSpeed * (double)floorNormalY) * ((double)startSpeed * (double)floorNormalY);
                                    double eqDet1 = eqB1 * eqB1 - 4.0 * eqA1 * eqC1;

                                    double eqA2 = ((double)q[1][0] - (double)q[0][0]) * ((double)q[1][0] - (double)q[0][0]) + ((double)q[1][2] - (double)q[0][2]) * ((double)q[1][2] - (double)q[0][2]);
                                    double eqB2 = 2.0 * (((double)q[1][0] - (double)q[0][0]) * ((double)q[0][0] - frame1Position[0]) + ((double)q[1][2] - (double)q[0][2]) * ((double)q[0][2] - frame1Position[2]));
                                    double eqC2 = ((double)q[0][0] - frame1Position[0]) * ((double)q[0][0] - frame1Position[0]) + ((double)q[0][2] - frame1Position[2]) * ((double)q[0][2] - frame1Position[2]) - ((double)startSpeed * (double)floorNormalY) * ((double)startSpeed * (double)floorNormalY);
                                    double eqDet2 = eqB2 * eqB2 - 4.0 * eqA2 * eqC2;

                                    if (eqDet1 >= 0 && eqDet2 >= 0) {
                                        double s = (-eqB1 + sqrt(eqDet1)) / (2.0 * eqA1);
                                        double t = (-eqB2 + sqrt(eqDet2)) / (2.0 * eqA2);

                                        if (s >= 0.0 && s <= 1.0 || t >= 0.0 && t <= 1.0) {
                                            intersectionIdxs[intersections] = squishEdges[i];
                                            startPositions[intersections][0][0] = ((double)p[1][0] - (double)p[0][0]) * s + (double)p[0][0];
                                            startPositions[intersections][0][1] = ((double)p[1][1] - (double)p[0][1]) * s + (double)p[0][1];
                                            startPositions[intersections][0][2] = ((double)p[1][2] - (double)p[0][2]) * s + (double)p[0][2];
                                            startPositions[intersections][1][0] = ((double)q[1][0] - (double)q[0][0]) * t + (double)q[0][0];
                                            startPositions[intersections][1][1] = ((double)q[1][1] - (double)q[0][1]) * t + (double)q[0][1];
                                            startPositions[intersections][1][2] = ((double)q[1][2] - (double)q[0][2]) * t + (double)q[0][2];

                                            adjust_position_to_ints(startPositions[intersections][0], startPositions[intersections][1], p);
                                            adjust_position_to_ints(startPositions[intersections][1], startPositions[intersections][0], q);

                                            intersections++;
                                            continue;
                                        }

                                        s = (-eqB1 - sqrt(eqDet1)) / (2.0 * eqA1);
                                        t = (-eqB2 - sqrt(eqDet2)) / (2.0 * eqA2);

                                        if (s >= 0.0 && s <= 1.0 || t >= 0.0 && t <= 1.0) {
                                            intersectionIdxs[intersections] = squishEdges[i];
                                            startPositions[intersections][0][0] = ((double)p[1][0] - (double)p[0][0]) * s + (double)p[0][0];
                                            startPositions[intersections][0][1] = ((double)p[1][1] - (double)p[0][1]) * s + (double)p[0][1];
                                            startPositions[intersections][0][2] = ((double)p[1][2] - (double)p[0][2]) * s + (double)p[0][2];
                                            startPositions[intersections][1][0] = ((double)q[1][0] - (double)q[0][0]) * t + (double)q[0][0];
                                            startPositions[intersections][1][1] = ((double)q[1][1] - (double)q[0][1]) * t + (double)q[0][1];
                                            startPositions[intersections][1][2] = ((double)q[1][2] - (double)q[0][2]) * t + (double)q[0][2];

                                            adjust_position_to_ints(startPositions[intersections][0], startPositions[intersections][1], p);
                                            adjust_position_to_ints(startPositions[intersections][1], startPositions[intersections][0], q);

                                            intersections++;
                                        }
                                    }
                                }
                            }

                            for (int i = 0; i < intersections; i++) {
                                float lowestPos = nextafterf(-2971.0f, INFINITY);
                                float highestPos = nextafterf(-2921.0f - (52.0f * sqrtf(1.0f - floorNormalY * floorNormalY) / floorNormalY), -INFINITY);

                                if (startPositions[i][0][1] < lowestPos) {
                                    if (startPositions[i][1][1] < lowestPos) {
                                        continue;
                                    }
                                    else {
                                        double t = (lowestPos - startPositions[i][0][1]) / (startPositions[i][1][1] - startPositions[i][0][1]);
                                        startPositions[i][0][0] = (startPositions[i][1][0] - startPositions[i][0][0]) * t + startPositions[i][0][0];
                                        startPositions[i][0][1] = (startPositions[i][1][1] - startPositions[i][0][1]) * t + startPositions[i][0][1];
                                        startPositions[i][0][2] = (startPositions[i][1][2] - startPositions[i][0][2]) * t + startPositions[i][0][2];
                                    }
                                }
                                else if (startPositions[i][1][1] < lowestPos) {
                                    double t = (lowestPos - startPositions[i][0][1]) / (startPositions[i][1][1] - startPositions[i][0][1]);
                                    startPositions[i][1][0] = (startPositions[i][1][0] - startPositions[i][0][0]) * t + startPositions[i][0][0];
                                    startPositions[i][1][1] = (startPositions[i][1][1] - startPositions[i][0][1]) * t + startPositions[i][0][1];
                                    startPositions[i][1][2] = (startPositions[i][1][2] - startPositions[i][0][2]) * t + startPositions[i][0][2];
                                }

                                if (startPositions[i][0][1] > highestPos) {
                                    if (startPositions[i][1][1] > highestPos) {
                                        continue;
                                    }
                                    else {
                                        double t = (highestPos - startPositions[i][0][1]) / (startPositions[i][1][1] - startPositions[i][0][1]);
                                        startPositions[i][0][0] = (startPositions[i][1][0] - startPositions[i][0][0]) * t + startPositions[i][0][0];
                                        startPositions[i][0][1] = (startPositions[i][1][1] - startPositions[i][0][1]) * t + startPositions[i][0][1];
                                        startPositions[i][0][2] = (startPositions[i][1][2] - startPositions[i][0][2]) * t + startPositions[i][0][2];
                                    }
                                }
                                else if (startPositions[i][1][1] > highestPos) {
                                    double t = (highestPos - startPositions[i][0][1]) / (startPositions[i][1][1] - startPositions[i][0][1]);
                                    startPositions[i][1][0] = (startPositions[i][1][0] - startPositions[i][0][0]) * t + startPositions[i][0][0];
                                    startPositions[i][1][1] = (startPositions[i][1][1] - startPositions[i][0][1]) * t + startPositions[i][0][1];
                                    startPositions[i][1][2] = (startPositions[i][1][2] - startPositions[i][0][2]) * t + startPositions[i][0][2];

                                }

                                int f1Angle0 = atan2sG(frame1Position[2] - startPositions[i][0][2], frame1Position[0] - startPositions[i][0][0]);
                                int f1Angle1 = atan2sG(frame1Position[2] - startPositions[i][1][2], frame1Position[0] - startPositions[i][1][0]);
                                f1Angle0 = (65536 + f1Angle0) % 65536;
                                f1Angle1 = (65536 + f1Angle1) % 65536;

                                if (f1Angle0 > f1Angle1) {
                                    int temp = f1Angle0;
                                    f1Angle0 = f1Angle1;
                                    f1Angle1 = temp;
                                }

                                if (f1Angle1 - f1Angle0 > 32768) {
                                    int temp = f1Angle0;
                                    f1Angle0 = f1Angle1;
                                    f1Angle1 = temp + 65536;
                                }

                                if (f1Angle0 <= sol5->f1Angle && f1Angle1 >= sol5->f1Angle) {
                                    float minBullyX = INFINITY;
                                    float maxBullyX = -INFINITY;
                                    float minBullyZ = INFINITY;
                                    float maxBullyZ = -INFINITY;
                                    int minPushAngle = INT_MAX;
                                    int maxPushAngle = INT_MIN;
                                    int refPushAngle = 65536;

                                    const float accel = 7.0f;
                                    const float pushRadius = 115.0f;
                                    const float bullyHurtbox = 63.0f;
                                    const float baseBullySpeed = powf(2.0f, 24);
                                    const float maxBullySpeed = nextafterf(powf(2.0f, 30), -INFINITY);

                                    int surfAngle = atan2sG(squishCeilingNormals[intersectionIdxs[i]][2], squishCeilingNormals[intersectionIdxs[i]][0]);
                                    surfAngle = (65536 + surfAngle) % 65536;

                                    float xPushVel = gSineTableG[surfAngle >> 4] * 10.0f;
                                    float zPushVel = gCosineTableG[surfAngle >> 4] * 10.0f;

                                    int squishFloorIdx = (intersectionIdxs[i] == 0 || intersectionIdxs[i] == 2) ? 0 : 1;

                                    int slopeAngle = atan2sG(startNormals[squishFloorIdx][2], startNormals[squishFloorIdx][0]);
                                    slopeAngle = (slopeAngle + 65536) % 65536;

                                    float steepness = sqrtf(startNormals[squishFloorIdx][0] * startNormals[squishFloorIdx][0] + startNormals[squishFloorIdx][2] * startNormals[squishFloorIdx][2]);

                                    float slopeXVel = accel * steepness * gSineTableG[slopeAngle >> 4];
                                    float slopeZVel = accel * steepness * gCosineTableG[slopeAngle >> 4];

                                    for (int j = 0; j < 2; j++) {
                                        float currentX = startPositions[i][j][0];
                                        float currentZ = startPositions[i][j][2];

                                        for (int k = 0; k < 3; k++) {
                                            float bullyPushX = currentX - xPushVel / 4.0f;
                                            float bullyPushZ = currentZ - zPushVel / 4.0f;

                                            int minAngle = INT_MAX;
                                            int maxAngle = INT_MIN;
                                            int refAngle = 65536;

                                            for (int l = 0; l < nSquishSpots[intersectionIdxs[i]]; l++) {
                                                float signX = (squishSpots[(2 * intersectionIdxs[i] * MAX_SQUISH_SPOTS) + (2 * l)] > 0) - (squishSpots[(2 * intersectionIdxs[i] * MAX_SQUISH_SPOTS) + (2 * l)] < 0);
                                                float signZ = (squishSpots[(2 * l) + 1] > 0) - (squishSpots[(2 * l) + 1] < 0);

                                                for (int m = 0; m < 4; m++) {
                                                    float xDist = bullyPushX - (squishSpots[(2 * intersectionIdxs[i] * MAX_SQUISH_SPOTS) + (2 * l)] + signX * (m % 2));
                                                    float zDist = bullyPushZ - (squishSpots[(2 * intersectionIdxs[i] * MAX_SQUISH_SPOTS) + (2 * l) + 1] + signZ * (m / 2));

                                                    float dist = sqrtf(xDist * xDist + zDist * zDist);

                                                    if (dist >= pushRadius - bullyHurtbox && dist <= pushRadius - fmaxf(bullyHurtbox - 2.0f * maxSlidingSpeed - 1.85f, 0.0f)) {
                                                        int angle = atan2sG(zDist, xDist);
                                                        angle = (angle + 65536) % 65536;

                                                        int angleDiff = (short)(angle - uphillAngle);

                                                        if (angleDiff < -0x4000 || angleDiff > 0x4000) {
                                                            if (refAngle == 65536) {
                                                                refAngle = angle;
                                                            }

                                                            minAngle = min(minAngle, (int)(short)(angle - refAngle));
                                                            maxAngle = max(maxAngle, (int)(short)(angle - refAngle));
                                                        }
                                                    }
                                                }
                                            }

                                            if (refAngle != 65536) {
                                                minAngle = (65536 + minAngle + refAngle) % 65536;
                                                maxAngle = (65536 + maxAngle + refAngle) % 65536;

                                                if (minAngle > maxAngle) {
                                                    minBullyX = fminf(minBullyX, bullyPushX - pushRadius);
                                                }
                                                else {
                                                    minBullyX = fminf(minBullyX, bullyPushX - pushRadius * gSineTableG[minAngle >> 4]);
                                                    minBullyX = fminf(minBullyX, bullyPushX - pushRadius * gSineTableG[maxAngle >> 4]);
                                                }

                                                if (minAngle < 16384 && maxAngle > 16384) {
                                                    minBullyZ = fminf(minBullyZ, bullyPushZ - pushRadius);
                                                }
                                                else {
                                                    minBullyZ = fminf(minBullyZ, bullyPushZ - pushRadius * gCosineTableG[minAngle >> 4]);
                                                    minBullyZ = fminf(minBullyZ, bullyPushZ - pushRadius * gCosineTableG[maxAngle >> 4]);
                                                }

                                                if (minAngle < 32768 && maxAngle > 32768) {
                                                    maxBullyX = fmaxf(maxBullyX, bullyPushX - pushRadius);
                                                }
                                                else {
                                                    maxBullyX = fmaxf(maxBullyX, bullyPushX - pushRadius * gSineTableG[minAngle >> 4]);
                                                    maxBullyX = fmaxf(maxBullyX, bullyPushX - pushRadius * gSineTableG[maxAngle >> 4]);
                                                }

                                                if (minAngle < 49152 && maxAngle > 49152) {
                                                    maxBullyZ = fmaxf(maxBullyZ, bullyPushZ - pushRadius);
                                                }
                                                else {
                                                    maxBullyZ = fmaxf(maxBullyZ, bullyPushZ - pushRadius * gCosineTableG[minAngle >> 4]);
                                                    maxBullyZ = fmaxf(maxBullyZ, bullyPushZ - pushRadius * gCosineTableG[maxAngle >> 4]);
                                                }

                                                if (refPushAngle == 65536) {
                                                    refPushAngle = minAngle;
                                                }

                                                minPushAngle = min(minPushAngle, (int)(minAngle - refAngle));
                                                maxPushAngle = max(maxPushAngle, (int)(short)(maxAngle - refAngle));
                                            }

                                            currentX = currentX - squishNormals[squishFloorIdx][1] * xPushVel / 4.0f;
                                            currentZ = currentZ - squishNormals[squishFloorIdx][1] * zPushVel / 4.0f;
                                        }
                                    }

                                    if (refPushAngle != 65536) {
                                        minPushAngle = (65536 + minPushAngle + refPushAngle) % 65536;
                                        maxPushAngle = (65536 + maxPushAngle + refPushAngle) % 65536;

                                        float xDiff2;

                                        if (minBullyX == maxBullyX) {
                                            int precision;
                                            frexpf(minBullyX, &precision);
                                            xDiff2 = powf(2.0f, precision - 24);
                                        }
                                        else {
                                            xDiff2 = powf(2.0f, floorf(log2f(maxBullyX - minBullyX)));

                                            while (floorf(maxBullyX / (2.0f * xDiff2)) >= ceilf(minBullyX / (2.0f * xDiff2))) {
                                                xDiff2 = xDiff2 * 2.0f;
                                            }
                                        }

                                        float zDiff2;

                                        if (minBullyZ == maxBullyZ) {
                                            int precision;
                                            frexpf(minBullyZ, &precision);
                                            zDiff2 = powf(2.0f, precision - 24);
                                        }
                                        else {
                                            zDiff2 = powf(2.0f, floorf(log2f(maxBullyZ - minBullyZ)));

                                            while (floorf(maxBullyZ / (2.0f * zDiff2)) >= ceilf(minBullyZ / (2.0f * zDiff2))) {
                                                zDiff2 = zDiff2 * 2.0f;
                                            }
                                        }

                                        float maxBullyXSpeed = fminf(nextafterf(xDiff2 * baseBullySpeed, -INFINITY), maxBullySpeed);
                                        float maxBullyZSpeed = fminf(nextafterf(zDiff2 * baseBullySpeed, -INFINITY), maxBullySpeed);

                                        float maxPushSpeed;

                                        int maxSpeedAngle = atan2sG(maxBullyXSpeed, maxBullyZSpeed);
                                        maxSpeedAngle = (65536 + maxSpeedAngle) % 65536;

                                        if ((65536 + maxSpeedAngle - minPushAngle) % 65536 < (65536 + maxPushAngle - minPushAngle) % 65536) {
                                            maxPushSpeed = (fabsf(maxBullyXSpeed * gSineTableG[maxSpeedAngle >> 4]) + fabsf(maxBullyZSpeed * gCosineTableG[maxSpeedAngle >> 4])) * (73.0f / 53.0f) * 3.0f;
                                        }
                                        else {
                                            float minAngleSpeed = (fabsf(maxBullyXSpeed * gSineTableG[minPushAngle >> 4]) + fabsf(maxBullyZSpeed * gCosineTableG[minPushAngle >> 4])) * (73.0f / 53.0f) * 3.0f;
                                            float maxAngleSpeed = (fabsf(maxBullyXSpeed * gSineTableG[maxPushAngle >> 4]) + fabsf(maxBullyZSpeed * gCosineTableG[maxPushAngle >> 4])) * (73.0f / 53.0f) * 3.0f;
                                            maxPushSpeed = fmaxf(minAngleSpeed, maxAngleSpeed);
                                        }

                                        float maxLossFactor = (-1.0 * (0.5f + 0.5f * maxPushSpeed / 100.0f)) * 0.02 + 0.92;

                                        float post10KXVel = (frame1Position[0] - ((startPositions[i][0][0] + startPositions[i][1][0]) / 2.0f)) / startNormals[squishFloorIdx][1];
                                        float post10KZVel = (frame1Position[2] - ((startPositions[i][0][2] + startPositions[i][1][2]) / 2.0f)) / startNormals[squishFloorIdx][1];

                                        float slidingSpeedX = (post10KXVel / maxLossFactor) - slopeXVel;
                                        float slidingSpeedZ = (post10KZVel / maxLossFactor) - slopeZVel;

                                        float slidingSpeedToPlatformOptions[4] = { -slidingSpeedX, slidingSpeedZ, -slidingSpeedZ, slidingSpeedX };

                                        float slidingSpeedToPlatform = slidingSpeedToPlatformOptions[intersectionIdxs[i]];

                                        if (fabsf(slidingSpeedX) <= maxSlidingSpeed && fabsf(slidingSpeedZ) <= maxSlidingSpeed && slidingSpeedToPlatform <= maxSlidingSpeedToPlatform) {
                                            int solIdx = atomicAdd(&n10KSolutions, 1);

                                            if (solIdx < MAX_10K_SOLUTIONS) {
                                                struct TenKSolution* solution = &(tenKSolutions[solIdx]);
                                                solution->speedSolutionIdx = idx;
                                                solution->departureSpeed = startSpeed;
                                                solution->pre10KVel[0] = pre10KVelX;
                                                solution->pre10KVel[1] = pre10KVelZ;
                                                solution->returnVel[0] = returnVelX;
                                                solution->returnVel[1] = returnVelZ;
                                                solution->frame2Position[0] = frame2Position[0];
                                                solution->frame2Position[1] = frame2Position[1];
                                                solution->frame2Position[2] = frame2Position[2];
                                                solution->frame1Position[0] = frame1Position[0];
                                                solution->frame1Position[1] = frame1Position[1];
                                                solution->frame1Position[2] = frame1Position[2];
                                                solution->startPosition[0][0] = startPositions[i][0][0];
                                                solution->startPosition[0][1] = startPositions[i][0][1];
                                                solution->startPosition[0][2] = startPositions[i][0][2];
                                                solution->startPosition[1][0] = startPositions[i][1][0];
                                                solution->startPosition[1][1] = startPositions[i][1][1];
                                                solution->startPosition[1][2] = startPositions[i][1][2];
                                                solution->squishCeiling = intersectionIdxs[i];
                                                solution->bdSetups = 0;
                                                solution->bpSetups = 0;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void find_speed_solutions() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strainIdx = idx % min(nStrainSetups, MAX_STRAIN_SETUPS);
    idx = idx / min(nStrainSetups, MAX_STRAIN_SETUPS);

    if (idx < min(nSKUWSolutions, MAX_SK_UPWARP_SOLUTIONS)) {
        struct StrainSetup* strain = &(strainSetups[strainIdx]);

        struct SKUpwarpSolution* sol = &(skuwSolutions[idx]);
        struct UpwarpSolution* uwSol = &(upwarpSolutions[sol->uwIdx]);
        struct PlatformSolution* platSol = &(platSolutions[uwSol->platformSolutionIdx]);

        struct SKPhase6* sol6 = &(sk6Solutions[sol->skIdx]);
        struct SKPhase5* sol5 = &(sk5Solutions[sol6->p5Idx]);
        struct SKPhase4* sol4 = &(sk4Solutions[sol5->p4Idx]);
        struct SKPhase3* sol3 = &(sk3Solutions[sol4->p3Idx]);
        struct SKPhase2* sol2 = (sol3->p2Type / 2 == 0) ? ((sol3->p2Type % 2 == 0) ? &(sk2ASolutions[sol3->p2Idx]) : &(sk2BSolutions[sol3->p2Idx])) : ((sol3->p2Type % 2 == 0) ? &(sk2CSolutions[sol3->p2Idx]) : &(sk2DSolutions[sol3->p2Idx]));
        struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);

        float signF = (strain->forwardStrain > 0) - (strain->forwardStrain < 0);
        float fStrain = strain->forwardStrain;
        float prevFStrain = strain->forwardStrain - signF*1.5f/(float)maxFSpeedLevels;

        fStrain = roundf(fStrain / sol->speedRange) * sol->speedRange;
        prevFStrain = roundf(prevFStrain / sol->speedRange) * sol->speedRange;

        float signS = (strain->sidewardStrain > 0) - (strain->sidewardStrain < 0);
        float prevSStrain = strain->sidewardStrain - signS * 10.0f / (float)maxSSpeedLevels;

        float xStrain = strain->sidewardStrain * gCosineTableG[sol2->f2Angle >> 4];
        float zStrain = strain->sidewardStrain * -gSineTableG[sol2->f2Angle >> 4];

        float prevXStrain = prevSStrain - signS * 1.5f / (float)maxSSpeedLevels;
        float prevZStrain = prevSStrain - signS * 1.5f / (float)maxSSpeedLevels;

        xStrain = roundf(xStrain / sol->xVelRange) * sol->xVelRange;
        zStrain = roundf(zStrain / sol->zVelRange) * sol->zVelRange;

        prevXStrain = roundf(prevXStrain / sol->xVelRange) * sol->xVelRange;
        prevZStrain = roundf(prevZStrain / sol->zVelRange) * sol->zVelRange;

        if ((strain->forwardStrain == 0.0f && strain->sidewardStrain == 0.0f) || fStrain != prevFStrain || xStrain != prevXStrain || zStrain != prevZStrain) {
            float minX = 65536.0f * sol3->x2 + tenKFloors[sol2->tenKFloorIdx][0];
            float maxX = 65536.0f * sol3->x2 + tenKFloors[sol2->tenKFloorIdx][1];
            float minZ = 65536.0f * sol3->z2 + tenKFloors[sol2->tenKFloorIdx][2];
            float maxZ = 65536.0f * sol3->z2 + tenKFloors[sol2->tenKFloorIdx][3];

            minX = nextafter(minX + (minX > 0) - (minX < 0), 0.0f);
            maxX = nextafter(maxX + (maxX > 0) - (maxX < 0), 0.0f);
            minZ = nextafter(minZ + (minZ > 0) - (minZ < 0), 0.0f);
            maxZ = nextafter(maxZ + (maxZ > 0) - (maxZ < 0), 0.0f);

            float minSpeed = sol->minSpeed;
            float maxSpeed = sol->maxSpeed;

            float minReturnVelX;
            float minReturnVelZ;
            float minPre10KSpeed = NAN;

            while (isnan(minPre10KSpeed) && minSpeed >= maxSpeed) {
                minPre10KSpeed = find_pre10K_speed(minSpeed, xStrain, zStrain, minReturnVelX, minReturnVelZ, sol->skIdx);

                if (isnan(minPre10KSpeed)) {
                    minSpeed = nextafterf(minSpeed, -INFINITY);
                }
            }

            float maxReturnVelX;
            float maxReturnVelZ;
            float maxPre10KSpeed = NAN;

            while (isnan(maxPre10KSpeed) && maxSpeed <= minSpeed) {
                maxPre10KSpeed = find_pre10K_speed(maxSpeed, xStrain, zStrain, maxReturnVelX, maxReturnVelZ, sol->skIdx);

                if (isnan(maxPre10KSpeed)) {
                    maxSpeed = nextafterf(maxSpeed, INFINITY);
                }
            }

            if (minSpeed >= maxSpeed) {
                float minSpeedF2X = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelX / 4.0f);
                float minSpeedF2Z = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelZ / 4.0f);

                float maxSpeedF2X = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (maxReturnVelX / 4.0f);
                float maxSpeedF2Z = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (maxReturnVelZ / 4.0f);

                bool speedTest = true;

                if (minSpeedF2X < minX) {
                    if (maxSpeedF2X < minX) {
                        speedTest = false;
                    }
                    else {
                        float lowerSpeed = minSpeed;
                        float upperSpeed = maxSpeed;

                        while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                            float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                            float testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, minReturnVelX, minReturnVelZ, sol->skIdx);

                            while (isnan(testPre10KSpeed) && midSpeed < lowerSpeed) {
                                midSpeed = nextafter(midSpeed, INFINITY);
                                testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, minReturnVelX, minReturnVelZ, sol->skIdx);
                            }

                            if (!isnan(testPre10KSpeed)) {
                                float newSpeedF2X = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelX / 4.0f);

                                if (newSpeedF2X < minX) {
                                    minPre10KSpeed = testPre10KSpeed;
                                    minSpeedF2X = newSpeedF2X;
                                    minSpeedF2Z = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelZ / 4.0f);
                                    lowerSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                                }
                                else {
                                    upperSpeed = midSpeed;
                                }
                            }
                            else {
                                lowerSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                            }
                        }

                        minSpeed = upperSpeed;
                    }
                }
                else if (maxSpeedF2X < minX) {
                    float lowerSpeed = minSpeed;
                    float upperSpeed = maxSpeed;

                    while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                        float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                        float testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, maxReturnVelX, maxReturnVelZ, sol->skIdx);

                        while (isnan(testPre10KSpeed) && midSpeed > upperSpeed) {
                            midSpeed = nextafter(midSpeed, -INFINITY);
                            testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, maxReturnVelX, maxReturnVelZ, sol->skIdx);
                        }

                        if (!isnan(testPre10KSpeed)) {
                            float newSpeedF2X = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (maxReturnVelX / 4.0f);

                            if (newSpeedF2X < minX) {
                                maxPre10KSpeed = testPre10KSpeed;
                                maxSpeedF2X = newSpeedF2X;
                                maxSpeedF2Z = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (maxReturnVelZ / 4.0f);
                                upperSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                            }
                            else {
                                lowerSpeed = midSpeed;
                            }
                        }
                        else {
                            upperSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                        }
                    }

                    maxSpeed = lowerSpeed;
                }

                if (speedTest) {
                    if (minSpeedF2X > maxX) {
                        if (maxSpeedF2X > maxX) {
                            speedTest = false;
                        }
                        else {
                            float lowerSpeed = minSpeed;
                            float upperSpeed = maxSpeed;

                            while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                                float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                                float testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, minReturnVelX, minReturnVelZ, sol->skIdx);

                                while (isnan(testPre10KSpeed) && midSpeed < lowerSpeed) {
                                    midSpeed = nextafter(midSpeed, INFINITY);
                                    testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, minReturnVelX, minReturnVelZ, sol->skIdx);
                                }

                                if (!isnan(testPre10KSpeed)) {
                                    float newSpeedF2X = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelX / 4.0f);

                                    if (newSpeedF2X > maxX) {
                                        minPre10KSpeed = testPre10KSpeed;
                                        minSpeedF2X = newSpeedF2X;
                                        minSpeedF2Z = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelZ / 4.0f);
                                        lowerSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                                    }
                                    else {
                                        upperSpeed = midSpeed;
                                    }
                                }
                                else {
                                    lowerSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                                }
                            }

                            minSpeed = upperSpeed;
                        }
                    }
                    else if (maxSpeedF2X > maxX) {
                        float lowerSpeed = minSpeed;
                        float upperSpeed = maxSpeed;

                        while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                            float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                            float testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, maxReturnVelX, maxReturnVelZ, sol->skIdx);

                            while (isnan(testPre10KSpeed) && midSpeed > upperSpeed) {
                                midSpeed = nextafter(midSpeed, -INFINITY);
                                testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, maxReturnVelX, maxReturnVelZ, sol->skIdx);
                            }

                            if (!isnan(testPre10KSpeed)) {
                                float newSpeedF2X = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (maxReturnVelX / 4.0f);

                                if (newSpeedF2X > maxX) {
                                    upperSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                                }
                                else {
                                    maxPre10KSpeed = testPre10KSpeed;
                                    maxSpeedF2X = newSpeedF2X;
                                    maxSpeedF2Z = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (maxReturnVelZ / 4.0f);
                                    lowerSpeed = midSpeed;
                                }
                            }
                            else {
                                upperSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                            }
                        }

                        minSpeed = lowerSpeed;
                    }
                }

                if (speedTest) {
                    if (minSpeedF2Z < minZ) {
                        if (maxSpeedF2Z < minZ) {
                            speedTest = false;
                        }
                        else {
                            float lowerSpeed = minSpeed;
                            float upperSpeed = maxSpeed;

                            while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                                float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                                float testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, minReturnVelX, minReturnVelZ, sol->skIdx);

                                while (isnan(testPre10KSpeed) && midSpeed < lowerSpeed) {
                                    midSpeed = nextafter(midSpeed, INFINITY);
                                    testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, minReturnVelX, minReturnVelZ, sol->skIdx);
                                }

                                if (!isnan(testPre10KSpeed)) {
                                    float newSpeedF2Z = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelZ / 4.0f);

                                    if (newSpeedF2Z < minZ) {
                                        lowerSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                                    }
                                    else {
                                        minPre10KSpeed = testPre10KSpeed;
                                        minSpeedF2Z = newSpeedF2Z;
                                        minSpeedF2X = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelX / 4.0f);
                                        upperSpeed = midSpeed;
                                    }
                                }
                                else {
                                    lowerSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                                }
                            }

                            minSpeed = upperSpeed;
                        }
                    }
                    else if (maxSpeedF2Z < minZ) {
                        float lowerSpeed = minSpeed;
                        float upperSpeed = maxSpeed;

                        while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                            float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                            float testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, maxReturnVelX, maxReturnVelZ, sol->skIdx);

                            while (isnan(testPre10KSpeed) && midSpeed > upperSpeed) {
                                midSpeed = nextafter(midSpeed, -INFINITY);
                                testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, maxReturnVelX, maxReturnVelZ, sol->skIdx);
                            }

                            if (!isnan(testPre10KSpeed)) {
                                float newSpeedF2Z = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (maxReturnVelZ / 4.0f);

                                if (newSpeedF2Z < minZ) {
                                    upperSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                                }
                                else {
                                    maxPre10KSpeed = testPre10KSpeed;
                                    maxSpeedF2Z = newSpeedF2Z;
                                    maxSpeedF2X = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (maxReturnVelX / 4.0f);
                                    lowerSpeed = midSpeed;
                                }
                            }
                            else {
                                upperSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                            }
                        }

                        maxSpeed = lowerSpeed;
                    }
                }

                if (speedTest) {
                    if (minSpeedF2Z > maxZ) {
                        if (maxSpeedF2Z > maxZ) {
                            speedTest = false;
                        }
                        else {
                            float lowerSpeed = minSpeed;
                            float upperSpeed = maxSpeed;

                            while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                                float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                                float testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, minReturnVelX, minReturnVelZ, sol->skIdx);

                                while (isnan(testPre10KSpeed) && midSpeed < lowerSpeed) {
                                    midSpeed = nextafter(midSpeed, INFINITY);
                                    testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, minReturnVelX, minReturnVelZ, sol->skIdx);
                                }

                                if (!isnan(testPre10KSpeed)) {
                                    float newSpeedF2Z = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelZ / 4.0f);

                                    if (newSpeedF2Z > maxZ) {
                                        lowerSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                                    }
                                    else {
                                        minPre10KSpeed = testPre10KSpeed;
                                        minSpeedF2Z = newSpeedF2Z;
                                        minSpeedF2X = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelX / 4.0f);
                                        upperSpeed = midSpeed;
                                    }
                                }
                                else {
                                    lowerSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                                }
                            }

                            minSpeed = upperSpeed;
                        }
                    }
                    else if (maxSpeedF2Z > maxZ) {
                        float lowerSpeed = minSpeed;
                        float upperSpeed = maxSpeed;

                        while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                            float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                            float testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, maxReturnVelX, maxReturnVelZ, sol->skIdx);

                            while (isnan(testPre10KSpeed) && midSpeed > upperSpeed) {
                                midSpeed = nextafter(midSpeed, -INFINITY);
                                testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, maxReturnVelX, maxReturnVelZ, sol->skIdx);
                            }

                            if (!isnan(testPre10KSpeed)) {
                                float newSpeedF2Z = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (maxReturnVelZ / 4.0f);

                                if (newSpeedF2Z > maxZ) {
                                    upperSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                                }
                                else {
                                    maxPre10KSpeed = testPre10KSpeed;
                                    maxSpeedF2Z = newSpeedF2Z;
                                    maxSpeedF2X = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (maxReturnVelX / 4.0f);
                                    lowerSpeed = midSpeed;
                                }
                            }
                            else {
                                upperSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                            }
                        }

                        maxSpeed = lowerSpeed;
                    }
                }

                float minPos[3] = { minSpeedF2X, 0.0f, minSpeedF2Z };
                float minFloorHeight;
                float maxPos[3] = { maxSpeedF2X, 0.0f, maxSpeedF2Z };
                float maxFloorHeight;
                SurfaceG* floor;

                if (speedTest) {
                    int minFloorIdx = find_floor(minPos, &floor, minFloorHeight, floorsG, total_floorsG);

                    if (minFloorIdx == -1 || floor->normal[1] != tenKFloors[sol2->tenKFloorIdx][7]) {
                        float lowerSpeed = minSpeed;
                        float upperSpeed = maxSpeed;

                        while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                            float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                            float testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, minReturnVelX, minReturnVelZ, sol->skIdx);

                            while (isnan(testPre10KSpeed) && midSpeed < lowerSpeed) {
                                midSpeed = nextafter(midSpeed, INFINITY);
                                testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, minReturnVelX, minReturnVelZ, sol->skIdx);
                            }

                            if (!isnan(testPre10KSpeed)) {
                                minPos[0] = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelX / 4.0f);
                                minPos[2] = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelZ / 4.0f);

                                SurfaceG* testFloor;
                                float testFloorHeight;
                                int testFloorIdx = find_floor(minPos, &testFloor, testFloorHeight, floorsG, total_floorsG);

                                if (testFloorIdx == -1 || testFloor->normal[1] != tenKFloors[sol2->tenKFloorIdx][7]) {
                                    lowerSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                                }
                                else {
                                    minFloorIdx = testFloorIdx;
                                    floor = testFloor;
                                    minFloorHeight = testFloorHeight;
                                    minPre10KSpeed = testPre10KSpeed;
                                    minSpeedF2X = minPos[0];
                                    minSpeedF2Z = minPos[2];
                                    upperSpeed = midSpeed;
                                }
                            }
                            else {
                                lowerSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                            }
                        }

                        minSpeed = upperSpeed;
                        speedTest = !(minFloorIdx == -1 && floor->normal[1] != tenKFloors[sol2->tenKFloorIdx][7]);
                    }
                }

                if (speedTest) {
                    int maxFloorIdx = find_floor(maxPos, &floor, maxFloorHeight, floorsG, total_floorsG);

                    if (maxFloorIdx == -1 || floor->normal[1] != tenKFloors[sol2->tenKFloorIdx][7]) {
                        float lowerSpeed = minSpeed;
                        float upperSpeed = maxSpeed;

                        while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                            float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                            float testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, maxReturnVelX, maxReturnVelZ, sol->skIdx);

                            while (isnan(testPre10KSpeed) && midSpeed > upperSpeed) {
                                midSpeed = nextafter(midSpeed, -INFINITY);
                                testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, maxReturnVelX, maxReturnVelZ, sol->skIdx);
                           }

                            if (!isnan(testPre10KSpeed)) {
                                maxPos[0] = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (maxReturnVelX / 4.0f);
                                maxPos[2] = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (maxReturnVelZ / 4.0f);

                                SurfaceG* testFloor;
                                float testFloorHeight;
                                int testFloorIdx = find_floor(maxPos, &testFloor, testFloorHeight, floorsG, total_floorsG);

                                if (testFloorIdx == -1 || testFloor->normal[1] != tenKFloors[sol2->tenKFloorIdx][7]) {
                                    upperSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                                }
                                else {
                                    maxFloorIdx = testFloorIdx;
                                    floor = testFloor;
                                    maxFloorHeight = testFloorHeight;
                                    maxPre10KSpeed = testPre10KSpeed;
                                    maxSpeedF2X = maxPos[0];
                                    maxSpeedF2Z = maxPos[2];
                                    lowerSpeed = midSpeed;
                                }
                            }
                            else {
                                upperSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                            }
                        }

                        maxSpeed = lowerSpeed;
                        speedTest = !(maxFloorIdx == -1 && floor->normal[1] != tenKFloors[sol2->tenKFloorIdx][7]);
                    }
                }

                
                if (speedTest) {
                    if (minFloorHeight > platSol->returnPosition[1]) {
                        if (maxFloorHeight > platSol->returnPosition[1]) {
                            speedTest = false;
                        }
                        else {
                            float lowerSpeed = minSpeed;
                            float upperSpeed = maxSpeed;

                            while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                                float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                                float testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, minReturnVelX, minReturnVelZ, sol->skIdx);

                                while (isnan(testPre10KSpeed) && midSpeed < lowerSpeed) {
                                    midSpeed = nextafter(midSpeed, INFINITY);
                                    testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, minReturnVelX, minReturnVelZ, sol->skIdx);
                                }

                                if (!isnan(testPre10KSpeed)) {
                                    minPos[0] = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelX / 4.0f);
                                    minPos[2] = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelZ / 4.0f);

                                    SurfaceG* testFloor;
                                    float testFloorHeight;
                                    int testFloorIdx = find_floor(minPos, &testFloor, testFloorHeight, floorsG, total_floorsG);

                                    if (testFloorIdx == -1 || testFloor->normal[1] != tenKFloors[sol2->tenKFloorIdx][7]) {
                                        break;
                                    } 
                                    else if (testFloorHeight > platSol->returnPosition[1]) {
                                        lowerSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                                    }
                                    else {
                                        minFloorHeight = testFloorHeight;
                                        minPre10KSpeed = testPre10KSpeed;
                                        minSpeedF2X = minPos[0];
                                        minSpeedF2Z = minPos[2];
                                        upperSpeed = midSpeed;
                                    }
                                }
                                else {
                                    lowerSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                                }
                            }

                            minSpeed = upperSpeed;
                            speedTest = !(minFloorHeight > platSol->returnPosition[1]);
                        }
                    }
                }

                if (speedTest) {
                    if (minFloorHeight <= -2971.0f || minFloorHeight <= (platSol->returnPosition[1] - 78.0f)) {
                        if (maxFloorHeight <= -2971.0f || maxFloorHeight <= (platSol->returnPosition[1] - 78.0f)) {
                            speedTest = false;
                        }
                        else {
                            float lowerSpeed = minSpeed;
                            float upperSpeed = maxSpeed;

                            while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                                float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                                float testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, minReturnVelX, minReturnVelZ, sol->skIdx);

                                while (isnan(testPre10KSpeed) && midSpeed < lowerSpeed) {
                                    midSpeed = nextafter(midSpeed, INFINITY);
                                    testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, minReturnVelX, minReturnVelZ, sol->skIdx);
                                }

                                if (!isnan(testPre10KSpeed)) {
                                    minPos[0] = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelX / 4.0f);
                                    minPos[2] = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelZ / 4.0f);

                                    SurfaceG* testFloor;
                                    float testFloorHeight;
                                    int testFloorIdx = find_floor(minPos, &testFloor, testFloorHeight, floorsG, total_floorsG);

                                    if (testFloorIdx == -1 || testFloor->normal[1] != tenKFloors[sol2->tenKFloorIdx][7]) {
                                        break;
                                    }
                                    else if (testFloorHeight <= -2971.0f || testFloorHeight <= (platSol->returnPosition[1] - 78.0f)) {
                                        lowerSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                                    }
                                    else {
                                        minFloorHeight = testFloorHeight;
                                        minPre10KSpeed = testPre10KSpeed;
                                        minSpeedF2X = minPos[0];
                                        minSpeedF2Z = minPos[2];
                                        upperSpeed = midSpeed;
                                    }
                                }
                                else {
                                    lowerSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                                }
                            }

                            minSpeed = upperSpeed;
                            speedTest = !(minFloorHeight <= -2971.0f || minFloorHeight <= (platSol->returnPosition[1] - 78.0f));
                        }
                    }
                }
                
                if (speedTest) {
                    if (maxFloorHeight > platSol->returnPosition[1]) {
                        float lowerSpeed = minSpeed;
                        float upperSpeed = maxSpeed;

                        while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                            float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                            float testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, maxReturnVelX, maxReturnVelZ, sol->skIdx);

                            while (isnan(testPre10KSpeed) && midSpeed > upperSpeed) {
                                midSpeed = nextafter(midSpeed, -INFINITY);
                                testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, maxReturnVelX, maxReturnVelZ, sol->skIdx);
                            }

                            if (!isnan(testPre10KSpeed)) {
                                maxPos[0] = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (maxReturnVelX / 4.0f);
                                maxPos[2] = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (maxReturnVelZ / 4.0f);

                                SurfaceG* testFloor;
                                float testFloorHeight;
                                int testFloorIdx = find_floor(maxPos, &testFloor, testFloorHeight, floorsG, total_floorsG);

                                if (testFloorIdx == -1 || testFloor->normal[1] != tenKFloors[sol2->tenKFloorIdx][7]) {
                                    break;
                                } 
                                else if (testFloorHeight > platSol->returnPosition[1]) {
                                    upperSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                                }
                                else {
                                    maxFloorHeight = testFloorHeight;
                                    maxPre10KSpeed = testPre10KSpeed;
                                    maxSpeedF2X = maxPos[0];
                                    maxSpeedF2Z = maxPos[2];
                                    lowerSpeed = midSpeed;
                                }
                            }
                            else {
                                upperSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                            }
                        }

                        maxSpeed = lowerSpeed;
                        speedTest = !(maxFloorHeight > platSol->returnPosition[1]);
                    }
                }

                if (speedTest) {
                    if (maxFloorHeight <= -2971.0f || maxFloorHeight <= (platSol->returnPosition[1] - 78.0f)) {
                        float lowerSpeed = minSpeed;
                        float upperSpeed = maxSpeed;

                        while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                            float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                            float testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, maxReturnVelX, maxReturnVelZ, sol->skIdx);

                            while (isnan(testPre10KSpeed) && midSpeed > upperSpeed) {
                                midSpeed = nextafter(midSpeed, -INFINITY);
                                testPre10KSpeed = find_pre10K_speed(midSpeed, xStrain, zStrain, maxReturnVelX, maxReturnVelZ, sol->skIdx);
                            }

                            if (!isnan(testPre10KSpeed)) {
                                maxPos[0] = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (maxReturnVelX / 4.0f);
                                maxPos[2] = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (maxReturnVelZ / 4.0f);

                                SurfaceG* testFloor;
                                float testFloorHeight;
                                int testFloorIdx = find_floor(maxPos, &testFloor, testFloorHeight, floorsG, total_floorsG);

                                if (testFloorIdx == -1 || testFloor->normal[1] != tenKFloors[sol2->tenKFloorIdx][7]) {
                                    break;
                                }
                                else if (testFloorHeight <= -2971.0f || testFloorHeight <= (platSol->returnPosition[1] - 78.0f)) {
                                    upperSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                                }
                                else {
                                    maxFloorHeight = testFloorHeight;
                                    maxPre10KSpeed = testPre10KSpeed;
                                    maxSpeedF2X = maxPos[0];
                                    maxSpeedF2Z = maxPos[2];
                                    lowerSpeed = midSpeed;
                                }
                            }
                            else {
                                upperSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));
                            }
                        }

                        maxSpeed = lowerSpeed;

                        speedTest = !(maxFloorHeight <= -2971.0f || maxFloorHeight <= (platSol->returnPosition[1] - 78.0f));
                    }
                }

                if (speedTest) {
                    for (float speed = minSpeed; speed >= maxSpeed; speed = nextafterf(speed, -INFINITY)) {
                        float returnVelX;
                        float returnVelZ;

                        float pre10KSpeed = find_pre10K_speed(speed, xStrain, zStrain, returnVelX, returnVelZ, sol->skIdx);

                        if (!isnan(pre10KSpeed)) {
                            int solIdx = atomicAdd(&nSpeedSolutions, 1);

                            if (solIdx < MAX_SPEED_SOLUTIONS) {
                                struct SpeedSolution* solution = &(speedSolutions[solIdx]);
                                solution->skuwSolutionIdx = idx;
                                solution->returnSpeed = speed;
                                solution->forwardStrain = fStrain;
                                solution->xStrain = xStrain;
                                solution->zStrain = zStrain;
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void find_sk_upwarp_solutions() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nUpwarpSolutions, MAX_UPWARP_SOLUTIONS)) {
        struct UpwarpSolution* uwSol = &(upwarpSolutions[idx]);
        struct PlatformSolution* platSol = &(platSolutions[uwSol->platformSolutionIdx]);

        float speedBuffer = 1000.0;

        double maxDist = -INFINITY;
        double minDist = INFINITY;

        for (int i = 0; i < 3; i++) {
            double xDist = 65536.0f * uwSol->pux + platSol->endTriangles[platSol->endFloorIdx][i][0] - platSol->endPosition[0];
            double zDist = 65536.0f * uwSol->puz + platSol->endTriangles[platSol->endFloorIdx][i][2] - platSol->endPosition[2];

            double dist = sqrt(xDist * xDist + zDist * zDist);

            minDist = fmin(dist, minDist);
            maxDist = fmax(dist, maxDist);
        }

        float upperSpeed = -(maxDist / platSol->endTriangleNormals[platSol->endFloorIdx][1]) / 0.9 - speedBuffer;
        float lowerSpeed = -(minDist / platSol->endTriangleNormals[platSol->endFloorIdx][1]) / 0.94 + speedBuffer;

        for (int i = 0; i < min(nSK6Solutions, MAX_SK_PHASE_SIX); i++) {
            SKPhase6* sk6Sol = &(sk6Solutions[i]);
            struct SKPhase5* sol5 = &(sk5Solutions[sk6Sol->p5Idx]);
            struct SKPhase4* sol4 = &(sk4Solutions[sol5->p4Idx]);
            struct SKPhase3* sol3 = &(sk3Solutions[sol4->p3Idx]);
            struct SKPhase2* sol2 = (sol3->p2Type / 2 == 0) ? ((sol3->p2Type % 2 == 0) ? &(sk2ASolutions[sol3->p2Idx]) : &(sk2BSolutions[sol3->p2Idx])) : ((sol3->p2Type % 2 == 0) ? &(sk2CSolutions[sol3->p2Idx]) : &(sk2DSolutions[sol3->p2Idx]));
            struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);
            
            float minSpeed = fminf(lowerSpeed, sk6Sol->minPost10KSpeed);
            float maxSpeed = fmaxf(upperSpeed, sk6Sol->maxPost10KSpeed);
            
            if (minSpeed >= maxSpeed) {
                float minReturnVelX;
                float minReturnVelZ;

                float minPre10KSpeed = NAN;

                while (isnan(minPre10KSpeed) && minSpeed >= maxSpeed) {
                    minPre10KSpeed = find_pre10K_speed(minSpeed, 0.0f, 0.0f, minReturnVelX, minReturnVelZ, i);

                    if (isnan(minPre10KSpeed)) {
                        minSpeed = nextafterf(minSpeed, -INFINITY);
                    }
                }

                if (minSpeed >= maxSpeed) {
                    int precision;

                    frexpf(minPre10KSpeed, &precision);

                    float speedRange = powf(2.0f, precision - 24);
                    int nFSpeedLevels = (int)ceilf(1.5f / speedRange);
                    atomicMax(&maxFSpeedLevels, nFSpeedLevels);

                    float xVel = minPre10KSpeed * gSineTableG[sol2->f2Angle >> 4];
                    float zVel = minPre10KSpeed * gCosineTableG[sol2->f2Angle >> 4];

                    frexpf(xVel, &precision);
                    float xVelRange = powf(2.0f, precision - 24);
                    int nXSpeedLevels = (int)ceilf(fabs(10.0f * gCosineTableG[sol2->f2Angle >> 4]) / xVelRange);

                    frexpf(zVel, &precision);
                    float zVelRange = powf(2.0f, precision - 24);
                    int nZSpeedLevels = (int)ceilf(fabs(10.0f * -gSineTableG[sol2->f2Angle >> 4]) / zVelRange);

                    atomicMax(&maxSSpeedLevels, max(nXSpeedLevels, nZSpeedLevels));

                    int solIdx = atomicAdd(&nSKUWSolutions, 1);

                    if (solIdx < MAX_SK_UPWARP_SOLUTIONS) {
                        struct SKUpwarpSolution* solution = &(skuwSolutions[solIdx]);
                        solution->skIdx = i;
                        solution->uwIdx = idx;
                        solution->minSpeed = minSpeed;
                        solution->maxSpeed = maxSpeed;
                        solution->speedRange = speedRange;
                        solution->xVelRange = xVelRange;
                        solution->zVelRange = zVelRange;
                    }
                }
            }
        }
    }
}

__device__ int calculate_camera_yaw(float* currentPosition, float* lakituPosition, int faceAngle) {
    short baseCameraYaw = -16384;
    float baseCameraDist = 1400.0;
    short baseCameraPitch = 0x05B0;

    SurfaceG* floor;
    float floorY;

    float xOff = currentPosition[0] + gSineTableG[((65536 + (int)baseCameraYaw) % 65536) >> 4] * 40.f;
    float zOff = currentPosition[2] + gCosineTableG[((65536 + (int)baseCameraYaw) % 65536) >> 4] * 40.f;
    float offPos[3] = { xOff, currentPosition[1], zOff };

    int floorIdx = find_floor(offPos, &floor, floorY, floorsG, total_floorsG);
    floorY = floorY - currentPosition[1];

    if (floorIdx != -1) {
        if (floorY > 0) {
            if (!(floor->normal[2] == 0.f && floorY < 100.f)) {
                baseCameraPitch += atan2sG(40.f, floorY);
            }
        }
    }

    baseCameraPitch = baseCameraPitch + 2304;

    float cameraPos[3] = { currentPosition[0] + baseCameraDist * gCosineTableG[((65536 + (int)baseCameraPitch) % 65536) >> 4] * gSineTableG[((65536 + (int)baseCameraYaw) % 65536) >> 4],
                       currentPosition[1] + 125.0f + baseCameraDist * gSineTableG[((65536 + (int)baseCameraPitch) % 65536) >> 4],
                       currentPosition[2] + baseCameraDist * gCosineTableG[((65536 + (int)baseCameraPitch) % 65536) >> 4] * gCosineTableG[((65536 + (int)baseCameraYaw) % 65536) >> 4]
    };

    float pan[3] = { 0, 0, 0 };
    float temp[3] = { 0, 0, 0 };

    // Get distance and angle from camera to Mario.
    float dx = currentPosition[0] - cameraPos[0];
    float dy = currentPosition[1] + 125.0f;
    float dz = currentPosition[2] - cameraPos[2];

    float cameraDist = sqrtf(dx * dx + dy * dy + dz * dz);
    float cameraPitch = atan2sG(sqrtf(dx * dx + dz * dz), dy);
    float cameraYaw = atan2sG(dz, dx);

    // The camera will pan ahead up to about 30% of the camera's distance to Mario.
    pan[2] = gSineTableG[0xC0] * cameraDist;

    temp[0] = pan[0];
    temp[1] = pan[1];
    temp[2] = pan[2];

    pan[0] = temp[2] * gSineTableG[((65536 + (int)faceAngle) % 65536) >> 4] + temp[0] * gCosineTableG[((65536 + (int)faceAngle) % 65536) >> 4];
    pan[2] = temp[2] * gCosineTableG[((65536 + (int)faceAngle) % 65536) >> 4] + temp[0] * gSineTableG[((65536 + (int)faceAngle) % 65536) >> 4];

    // rotate in the opposite direction
    cameraYaw = -cameraYaw;

    temp[0] = pan[0];
    temp[1] = pan[1];
    temp[2] = pan[2];

    pan[0] = temp[2] * gSineTableG[((65536 + (int)cameraYaw) % 65536) >> 4] + temp[0] * gCosineTableG[((65536 + (int)cameraYaw) % 65536) >> 4];
    pan[2] = temp[2] * gCosineTableG[((65536 + (int)cameraYaw) % 65536) >> 4] + temp[0] * gSineTableG[((65536 + (int)cameraYaw) % 65536) >> 4];

    // Only pan left or right
    pan[2] = 0.f;

    cameraYaw = -cameraYaw;

    temp[0] = pan[0];
    temp[1] = pan[1];
    temp[2] = pan[2];

    pan[0] = temp[2] * gSineTableG[((65536 + (int)cameraYaw) % 65536) >> 4] + temp[0] * gCosineTableG[((65536 + (int)cameraYaw) % 65536) >> 4];
    pan[2] = temp[2] * gCosineTableG[((65536 + (int)cameraYaw) % 65536) >> 4] + temp[0] * gSineTableG[((65536 + (int)cameraYaw) % 65536) >> 4];

    float cameraFocus[3] = { currentPosition[0] + pan[0], currentPosition[1] + 125.0f + pan[1], currentPosition[2] + pan[2] };

    dx = cameraFocus[0] - lakituPosition[0];
    dy = cameraFocus[1] - lakituPosition[1];
    dz = cameraFocus[2] - lakituPosition[2];

    cameraDist = sqrtf(dx * dx + dy * dy + dz * dz);
    cameraPitch = atan2sG(sqrtf(dx * dx + dz * dz), dy);
    cameraYaw = atan2sG(dz, dx);

    if (cameraPitch > 15872) {
        cameraPitch = 15872;
    }
    if (cameraPitch < -15872) {
        cameraPitch = -15872;
    }

    cameraFocus[0] = lakituPosition[0] + cameraDist * gCosineTableG[((65536 + (int)cameraPitch) % 65536) >> 4] * gSineTableG[((65536 + (int)cameraYaw) % 65536) >> 4];
    cameraFocus[1] = lakituPosition[1] + cameraDist * gSineTableG[((65536 + (int)cameraPitch) % 65536) >> 4];
    cameraFocus[2] = lakituPosition[2] + cameraDist * gCosineTableG[((65536 + (int)cameraPitch) % 65536) >> 4] * gCosineTableG[((65536 + (int)cameraYaw) % 65536) >> 4];

    return atan2sG(lakituPosition[2] - cameraFocus[2], lakituPosition[0] - cameraFocus[0]);
}

__device__ void platform_logic(float* platform_normal, float* mario_pos, short(&triangles)[2][3][3], float(&normals)[2][3], float(&mat)[4][4]) {
    float dx;
    float dy;
    float dz;
    float d;

    float dist[3];
    float posBeforeRotation[3];
    float posAfterRotation[3];

    // Mario's position
    float mx = mario_pos[0];
    float my = mario_pos[1];
    float mz = mario_pos[2];

    dist[0] = mx - (float)platform_pos[0];
    dist[1] = my - (float)platform_pos[1];
    dist[2] = mz - (float)platform_pos[2];

    mat[1][0] = platform_normal[0];
    mat[1][1] = platform_normal[1];
    mat[1][2] = platform_normal[2];

    float invsqrt = 1.0f / sqrtf(mat[1][0] * mat[1][0] + mat[1][1] * mat[1][1] + mat[1][2] * mat[1][2]);

    mat[1][0] *= invsqrt;
    mat[1][1] *= invsqrt;
    mat[1][2] *= invsqrt;

    mat[0][0] = mat[1][1] * 1.0f - 0.0f * mat[1][2];
    mat[0][1] = mat[1][2] * 0.0f - 1.0f * mat[1][0];
    mat[0][2] = mat[1][0] * 0.0f - 0.0f * mat[1][1];

    invsqrt = 1.0f / sqrtf(mat[0][0] * mat[0][0] + mat[0][1] * mat[0][1] + mat[0][2] * mat[0][2]);

    mat[0][0] *= invsqrt;
    mat[0][1] *= invsqrt;
    mat[0][2] *= invsqrt;

    mat[2][0] = mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2];
    mat[2][1] = mat[0][2] * mat[1][0] - mat[1][2] * mat[0][0];
    mat[2][2] = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];

    invsqrt = 1.0f / sqrtf(mat[2][0] * mat[2][0] + mat[2][1] * mat[2][1] + mat[2][2] * mat[2][2]);

    mat[2][0] *= invsqrt;
    mat[2][1] *= invsqrt;
    mat[2][2] *= invsqrt;

    mat[3][0] = platform_pos[0];
    mat[3][1] = platform_pos[1];
    mat[3][2] = platform_pos[2];
    mat[0][3] = 0.0f;
    mat[1][3] = 0.0f;
    mat[2][3] = 0.0f;
    mat[3][3] = 1.0f;

    for (int i = 0; i < 3; i++) {
        posBeforeRotation[i] = mat[0][i] * dist[0] + mat[1][i] * dist[1] + mat[2][i] * dist[2];
    }

    dx = mx - (float)platform_pos[0];
    dy = 500.0f;
    dz = mz - (float)platform_pos[2];
    d = sqrtf(dx * dx + dy * dy + dz * dz);

    // Normalizing
    d = 1.0 / d;
    dx *= d;
    dy *= d;
    dz *= d;

    // Approach the normals by 0.01f towards the new goal, then create a transform matrix and orient the object. 
    // Outside of the other conditionals since it needs to tilt regardless of whether Mario is on.
    platform_normal[0] = (platform_normal[0] <= dx) ? ((dx - platform_normal[0] < 0.01f) ? dx : (platform_normal[0] + 0.01f)) : ((dx - platform_normal[0] > -0.01f) ? dx : (platform_normal[0] - 0.01f));
    platform_normal[1] = (platform_normal[1] <= dy) ? ((dy - platform_normal[1] < 0.01f) ? dy : (platform_normal[1] + 0.01f)) : ((dy - platform_normal[1] > -0.01f) ? dy : (platform_normal[1] - 0.01f));
    platform_normal[2] = (platform_normal[2] <= dz) ? ((dz - platform_normal[2] < 0.01f) ? dz : (platform_normal[2] + 0.01f)) : ((dz - platform_normal[2] > -0.01f) ? dz : (platform_normal[2] - 0.01f));

    mat[1][0] = platform_normal[0];
    mat[1][1] = platform_normal[1];
    mat[1][2] = platform_normal[2];

    invsqrt = 1.0f / sqrtf(mat[1][0] * mat[1][0] + mat[1][1] * mat[1][1] + mat[1][2] * mat[1][2]);

    mat[1][0] *= invsqrt;
    mat[1][1] *= invsqrt;
    mat[1][2] *= invsqrt;

    mat[0][0] = mat[1][1] * 1.0f - 0.0f * mat[1][2];
    mat[0][1] = mat[1][2] * 0.0f - 1.0f * mat[1][0];
    mat[0][2] = mat[1][0] * 0.0f - 0.0f * mat[1][1];

    invsqrt = 1.0f / sqrtf(mat[0][0] * mat[0][0] + mat[0][1] * mat[0][1] + mat[0][2] * mat[0][2]);

    mat[0][0] *= invsqrt;
    mat[0][1] *= invsqrt;
    mat[0][2] *= invsqrt;

    mat[2][0] = mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2];
    mat[2][1] = mat[0][2] * mat[1][0] - mat[1][2] * mat[0][0];
    mat[2][2] = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];

    invsqrt = 1.0f / sqrtf(mat[2][0] * mat[2][0] + mat[2][1] * mat[2][1] + mat[2][2] * mat[2][2]);

    mat[2][0] *= invsqrt;
    mat[2][1] *= invsqrt;
    mat[2][2] *= invsqrt;

    mat[3][0] = platform_pos[0];
    mat[3][1] = platform_pos[1];
    mat[3][2] = platform_pos[2];
    mat[0][3] = 0.0f;
    mat[1][3] = 0.0f;
    mat[2][3] = 0.0f;
    mat[3][3] = 1.0f;

    for (int i = 0; i < 3; i++) {
        posAfterRotation[i] = mat[0][i] * dist[0] + mat[1][i] * dist[1] + mat[2][i] * dist[2];
    }

    mx += posAfterRotation[0] - posBeforeRotation[0];
    my += posAfterRotation[1] - posBeforeRotation[1];
    mz += posAfterRotation[2] - posBeforeRotation[2];
    mario_pos[0] = mx;
    mario_pos[1] = my;
    mario_pos[2] = mz;
}

__global__ void find_breakdance_solutions() {
    long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = (idx % 242) - 121;
    idx = idx / 242;
    int y = (idx % 242) - 121;
    idx = idx / 242;

    if (idx < min(nSlideSolutions, MAX_SLIDE_SOLUTIONS)) {
        struct SlideSolution* slideSol = &(slideSolutions[idx]);
        struct TenKSolution* tenKSol = &(tenKSolutions[slideSol->tenKSolutionIdx]);
        
        SurfaceG* floor;
        float floorHeight;

        int floorIdx = find_floor(slideSol->upwarpPosition, &floor, floorHeight, floorsG, total_floorsG);

        int slopeAngle = atan2sG(floor->normal[2], floor->normal[0]);
        slopeAngle = (slopeAngle + 65536) % 65536;

        float steepness = sqrtf(floor->normal[0] * floor->normal[0] + floor->normal[2] * floor->normal[2]);

        float cameraPositions[4][3] = { {-8192, -2918, -8192}, {-8192, -2918, 8191}, {8191, -2918, -8192}, {8191, -2918, 8191} };

        int minCameraYaw = 0;
        int maxCameraYaw = 0;

        int refCameraYaw = calculate_camera_yaw(slideSol->upwarpPosition, cameraPositions[0], slideSol->postSlideAngle);
        refCameraYaw = (65536 + refCameraYaw) % 65536;

        for (int i = 1; i < 4; i++) {
            int cameraYaw = calculate_camera_yaw(slideSol->upwarpPosition, cameraPositions[i], slideSol->postSlideAngle);
            cameraYaw = (short)(cameraYaw - refCameraYaw);
            minCameraYaw = min(minCameraYaw, cameraYaw);
            maxCameraYaw = max(maxCameraYaw, cameraYaw);
        }

        int minCameraIdx = gReverseArctanTableG[(65536 + minCameraYaw + refCameraYaw) % 65536];
        int maxCameraIdx = gReverseArctanTableG[(65536 + maxCameraYaw + refCameraYaw) % 65536];

        if (minCameraIdx > maxCameraIdx) {
            maxCameraIdx += 8192;
        }

        for (int i = minCameraIdx; i <= maxCameraIdx; i++) {
            int cameraYaw = gArctanTableG[i % 8192];
            cameraYaw = (65536 + cameraYaw) % 65536;

            if (validCameraAngle[cameraYaw]) {
                float stickX = x - (x < 0) + (x > 0);
                float stickY = y - (y < 0) + (y > 0);

                int rawX = x - 7 * (x < 0) + 7 * (x > 0);
                int rawY = y - 7 * (y < 0) + 7 * (y > 0);

                float stickMag = sqrtf(stickX * stickX + stickY * stickY);

                if (stickMag > 64) {
                    stickX *= 64 / stickMag;
                    stickY *= 64 / stickMag;
                    stickMag = 64;
                }

                float intendedMag = ((stickMag / 64.0f) * (stickMag / 64.0f)) * 32.0f;

                int intendedYaw;

                if (intendedMag > 0.0f) {
                    intendedYaw = atan2sG(-stickY, stickX) + cameraYaw;
                    intendedYaw = (65536 + intendedYaw) % 65536;
                }
                else {
                    intendedYaw = slideSol->postSlideAngle;
                }

                int intendedDYaw = (short)(intendedYaw - slideSol->postSlideAngle);
                intendedDYaw = (65536 + intendedDYaw) % 65536;

                float lossFactor = intendedMag / 32.0f * gCosineTableG[intendedDYaw >> 4] * 0.02f + 0.92f;

                float xVel = slideSol->postSlideSpeed * gSineTableG[slideSol->postSlideAngle >> 4];
                float zVel = slideSol->postSlideSpeed * gCosineTableG[slideSol->postSlideAngle >> 4];

                float oldSpeed = sqrtf(xVel * xVel + zVel * zVel);

                xVel += zVel * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;
                zVel -= xVel * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;

                float newSpeed = sqrtf(xVel * xVel + zVel * zVel);

                xVel = xVel * oldSpeed / newSpeed;
                zVel = zVel * oldSpeed / newSpeed;

                xVel += 7.0f * steepness * gSineTableG[slopeAngle >> 4];
                zVel += 7.0f * steepness * gCosineTableG[slopeAngle >> 4];

                xVel *= lossFactor;
                zVel *= lossFactor;

                float intendedPos[3] = { slideSol->upwarpPosition[0], slideSol->upwarpPosition[1], slideSol->upwarpPosition[2] };
                SurfaceG* newFloor = floor;
                bool fallTest = false;

                for (int j = 0; j < 4; j++) {
                    intendedPos[0] = intendedPos[0] + newFloor->normal[1] * (xVel / 4.0f);
                    intendedPos[2] = intendedPos[2] + newFloor->normal[1] * (zVel / 4.0f);

                    int floorIdx = find_floor(intendedPos, &newFloor, floorHeight, floorsG, total_floorsG);

                    if (floorIdx == -1) {
                        break;
                    }
                    else if (intendedPos[1] > floorHeight + 100.0f) {
                        fallTest = true;
                        break;
                    }
                }

                if (fallTest) {
                    int slideYaw = atan2sG(zVel, xVel);
                    slideYaw = (65536 + slideYaw) % 65536;

                    int facingDYaw = slideSol->postSlideAngle - slideYaw;

                    int newFacingDYaw = (short)facingDYaw;

                    if (newFacingDYaw > 0 && newFacingDYaw <= 0x4000) {
                        if ((newFacingDYaw -= 0x200) < 0) {
                            newFacingDYaw = 0;
                        }
                    }
                    else if (newFacingDYaw > -0x4000 && newFacingDYaw < 0) {
                        if ((newFacingDYaw += 0x200) > 0) {
                            newFacingDYaw = 0;
                        }
                    }
                    else if (newFacingDYaw > 0x4000 && newFacingDYaw < 0x8000) {
                        if ((newFacingDYaw += 0x200) > 0x8000) {
                            newFacingDYaw = 0x8000;
                        }
                    }
                    else if (newFacingDYaw > -0x8000 && newFacingDYaw < -0x4000) {
                        if ((newFacingDYaw -= 0x200) < -0x8000) {
                            newFacingDYaw = -0x8000;
                        }
                    }

                    int postSlideAngle = slideYaw + newFacingDYaw;
                    postSlideAngle = (65536 + postSlideAngle) % 65536;

                    float postSlideSpeed = -sqrtf(xVel * xVel + zVel * zVel);

                    xVel = postSlideSpeed * gSineTableG[postSlideAngle >> 4];
                    float yVel = 0.0f;
                    zVel = postSlideSpeed * gCosineTableG[postSlideAngle >> 4];

                    bool falling = true;
                    bool landed = false;

                    while (falling) {
                        for (int j = 0; j < 4; j++) {
                            intendedPos[0] = intendedPos[0] + (xVel / 4.0f);
                            intendedPos[1] = intendedPos[1] + (yVel / 4.0f);
                            intendedPos[2] = intendedPos[2] + (zVel / 4.0f);

                            float oldFloorHeight = floorHeight;
                            int floorIdx = find_floor(intendedPos, &newFloor, floorHeight, floorsG, total_floorsG);

                            if (floorIdx == -1) {
                                if (intendedPos[1] <= oldFloorHeight) {
                                    intendedPos[1] = oldFloorHeight;
                                    landed = true;
                                }
                                falling = false;
                                break;
                            }
                            else if (newFloor->normal[1] < 0.7880108) {
                                falling = false;
                                break;
                            }
                            else if (intendedPos[1] <= floorHeight) {
                                yVel = 0.0f;
                                intendedPos[1] = floorHeight;
                                falling = false;

                                if (!newFloor->is_lava) {
                                    float nextPos[3] = { intendedPos[0], intendedPos[1], intendedPos[2] };

                                    for (int k = 0; k < 12; k++) {
                                        nextPos[0] = nextPos[0] + newFloor->normal[1] * (xVel / 4.0f);
                                        nextPos[2] = nextPos[2] + newFloor->normal[1] * (zVel / 4.0f);

                                        floorIdx = find_floor(nextPos, &newFloor, floorHeight, floorsG, total_floorsG);

                                        if (floorIdx == -1) {
                                            landed = true;
                                            break;
                                        }
                                        else {
                                            intendedPos[0] = nextPos[0];
                                            intendedPos[2] = nextPos[2];
                                            
                                            if (intendedPos[1] > floorHeight + 100.0f) {
                                                falling = true;
                                                break;
                                            }
                                            else {
                                                intendedPos[1] = floorHeight;
                                            }
                                        }
                                    }

                                    if (falling) {
                                        continue;
                                    }
                                }

                                break;
                            }
                            else if (intendedPos[1] < -1357.0) {
                                falling = false;
                                break;
                            }
                            else if (intendedPos[0] < INT_MIN || intendedPos[0] > INT_MAX || intendedPos[2] < INT_MIN || intendedPos[2] > INT_MAX) {
                                falling = false;
                                break;
                            }
                        }

                        yVel = fminf(yVel - 4.0f, 75.0f);
                    }

                    if (landed && intendedPos[1] >= -1357.0) {
                        int solIdx = atomicAdd(&nBDSolutions, 1);

                        if (solIdx < MAX_BD_SOLUTIONS) {
                            BDSolution* solution = &(bdSolutions[solIdx]);
                            solution->slideSolutionIdx = idx;
                            solution->cameraYaw = cameraYaw;
                            solution->stickX = rawX;
                            solution->stickY = rawY;
                            solution->landingPosition[0] = intendedPos[0];
                            solution->landingPosition[1] = intendedPos[1];
                            solution->landingPosition[2] = intendedPos[2];
                            solution->postSlideSpeed = postSlideSpeed;
                            atomicAdd(&(tenKSol->bdSetups), 1);
                        }
                    }
                }
            }
        }
    }
}

__device__ void try_upwarp_slide(int solIdx, int angle, int intendedDYaw, float intendedMag) {
    struct TenKSolution* tenKSol = &(tenKSolutions[solIdx]);
    struct SpeedSolution* speedSol = &(speedSolutions[tenKSol->speedSolutionIdx]);
    struct SKUpwarpSolution* skuwSol = &(skuwSolutions[speedSol->skuwSolutionIdx]);
    struct UpwarpSolution* uwSol = &(upwarpSolutions[skuwSol->uwIdx]);
    struct PlatformSolution* platSol = &(platSolutions[uwSol->platformSolutionIdx]);

    float lossFactor = intendedMag / 32.0f * gCosineTableG[intendedDYaw >> 4] * 0.02f + 0.92f;
    
    int slopeAngle = atan2sG(platSol->endTriangleNormals[platSol->endFloorIdx][2], platSol->endTriangleNormals[platSol->endFloorIdx][0]);
    slopeAngle = (65536 + slopeAngle) % 65536;

    float steepness = sqrtf(platSol->endTriangleNormals[platSol->endFloorIdx][0] * platSol->endTriangleNormals[platSol->endFloorIdx][0] + platSol->endTriangleNormals[platSol->endFloorIdx][2] * platSol->endTriangleNormals[platSol->endFloorIdx][2]);

    float xVel0 = speedSol->returnSpeed * gSineTableG[angle >> 4];
    float zVel0 = speedSol->returnSpeed * gCosineTableG[angle >> 4];

    float xVel1 = xVel0;
    float zVel1 = zVel0;

    float oldSpeed = sqrtf(xVel1 * xVel1 + zVel1 * zVel1);

    xVel1 += zVel1 * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;
    zVel1 -= xVel1 * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;

    float newSpeed = sqrtf(xVel1 * xVel1 + zVel1 * zVel1);

    xVel1 = xVel1 * oldSpeed / newSpeed;
    zVel1 = zVel1 * oldSpeed / newSpeed;

    xVel1 += 7.0f * steepness * gSineTableG[slopeAngle >> 4];
    zVel1 += 7.0f * steepness * gCosineTableG[slopeAngle >> 4];

    xVel1 *= lossFactor;
    zVel1 *= lossFactor;

    float intendedPos[3] = { platSol->endPosition[0], platSol->endPosition[1], platSol->endPosition[2] };

    int floorIdx = platSol->endFloorIdx;
    bool slideCheck = true;

    for (int s = 0; s < 4; s++) {
        intendedPos[0] = intendedPos[0] + platSol->endTriangleNormals[floorIdx][1] * (xVel1 / 4.0f);
        intendedPos[2] = intendedPos[2] + platSol->endTriangleNormals[floorIdx][1] * (zVel1 / 4.0f);

        float floorHeight;
        floorIdx = find_floor(intendedPos, platSol->endTriangles, platSol->endTriangleNormals, &floorHeight);

        if (floorIdx == -1 || floorHeight <= -3071.0f) {
            slideCheck = false;
            break;
        }
        else {
            intendedPos[1] = floorHeight;
        }
    }

    if (slideCheck) {
        float prePositionTest[3] = { platSol->penultimatePosition[0] + platSol->penultimateFloorNormalY * xVel0 / 4.0f, platSol->penultimatePosition[1], platSol->penultimatePosition[2] + platSol->penultimateFloorNormalY * zVel0 / 4.0f };

        if (!check_inbounds(prePositionTest)) {
            float test_normal[3] = { platSol->endNormal[0], platSol->endNormal[1], platSol->endNormal[2] };
            float mario_pos[3] = { intendedPos[0], intendedPos[1], intendedPos[2]};

            short triangles[2][3][3];
            float normals[2][3];
            float mat[4][4];

            platform_logic(test_normal, mario_pos, triangles, normals, mat);

            bool upwarpPositionTest = false;

            for (int i = 0; i < n_y_ranges && !upwarpPositionTest; i++) {
                if (mario_pos[1] >= lower_y[i] && mario_pos[1] <= upper_y[i]) {
                    upwarpPositionTest = true;
                }
            }

            upwarpPositionTest = upwarpPositionTest && check_inbounds(mario_pos);

            if (upwarpPositionTest) {
                int idx = atomicAdd(&nSlideSolutions, 1);

                if (idx < MAX_SLIDE_SOLUTIONS) {
                    int slideYaw = atan2sG(zVel1, xVel1);
                    slideYaw = (65536 + slideYaw) % 65536;

                    int facingDYaw = angle - slideYaw;

                    int newFacingDYaw = (short)facingDYaw;

                    if (newFacingDYaw > 0 && newFacingDYaw <= 0x4000) {
                        if ((newFacingDYaw -= 0x200) < 0) {
                            newFacingDYaw = 0;
                        }
                    }
                    else if (newFacingDYaw > -0x4000 && newFacingDYaw < 0) {
                        if ((newFacingDYaw += 0x200) > 0) {
                            newFacingDYaw = 0;
                        }
                    }
                    else if (newFacingDYaw > 0x4000 && newFacingDYaw < 0x8000) {
                        if ((newFacingDYaw += 0x200) > 0x8000) {
                            newFacingDYaw = 0x8000;
                        }
                    }
                    else if (newFacingDYaw > -0x8000 && newFacingDYaw < -0x4000) {
                        if ((newFacingDYaw -= 0x200) < -0x8000) {
                            newFacingDYaw = -0x8000;
                        }
                    }

                    int postSlideAngle = slideYaw + newFacingDYaw;
                    postSlideAngle = (65536 + postSlideAngle) % 65536;

                    float postSlideSpeed = -sqrtf(xVel1 * xVel1 + zVel1 * zVel1);

                    SlideSolution* solution = &(slideSolutions[idx]);
                    solution->tenKSolutionIdx = solIdx;
                    solution->preUpwarpPosition[0] = intendedPos[0];
                    solution->preUpwarpPosition[1] = intendedPos[1];
                    solution->preUpwarpPosition[2] = intendedPos[2];
                    solution->upwarpPosition[0] = mario_pos[0];
                    solution->upwarpPosition[1] = mario_pos[1];
                    solution->upwarpPosition[2] = mario_pos[2];
                    solution->angle = angle;
                    solution->intendedDYaw = intendedDYaw;
                    solution->stickMag = intendedMag;
                    solution->postSlideAngle = postSlideAngle;
                    solution->postSlideSpeed = postSlideSpeed;
                }
            }
        }
    }
}

__device__ void try_pu_slide_angle(int solIdx, int angle, double minEndAngle, double maxEndAngle, double minM1, double maxM1) {
    double minAngleDiff = fmax(minEndAngle - angle, -(double)522);
    double maxAngleDiff = fmin(maxEndAngle - angle, (double)522);

    if (minAngleDiff <= maxAngleDiff) {
        double minEndAngleA = minAngleDiff + angle;
        double maxEndAngleA = maxAngleDiff + angle;

        double minN;
        double maxN;

        if (angle == 0 || angle == 32768) {
            double sinStartAngle = sin(2.0 * M_PI * (double)angle / 65536.0);

            minN = -cos(2.0 * M_PI * minEndAngleA / 65536.0) / sinStartAngle;
            maxN = -cos(2.0 * M_PI * maxEndAngleA / 65536.0) / sinStartAngle;
        }
        else {
            double sinStartAngle = gSineTableG[angle >> 4];
            double cosStartAngle = gCosineTableG[angle >> 4];

            double sinMinEndAngle = sin(2.0 * M_PI * minEndAngleA / 65536.0);
            double cosMinEndAngle = cos(2.0 * M_PI * minEndAngleA / 65536.0);

            double sinMaxEndAngle = sin(2.0 * M_PI * maxEndAngleA / 65536.0);
            double cosMaxEndAngle = cos(2.0 * M_PI * maxEndAngleA / 65536.0);

            double t = sinStartAngle / cosStartAngle;
            double s = sinMinEndAngle / cosMinEndAngle;

            bool signTest = (cosStartAngle > 0 && cosMinEndAngle > 0) || (cosStartAngle < 0 && cosMinEndAngle < 0);

            if (signTest) {
                minN = (-((double)s * (double)t) - 1.0 + sqrt(((double)s * (double)t - 1.0) * ((double)s * (double)t - 1.0) + 4.0 * (double)s * (double)s)) / (2.0 * (double)s);
            }
            else {
                minN = (-((double)s * (double)t) - 1.0 - sqrt(((double)s * (double)t - 1.0) * ((double)s * (double)t - 1.0) + 4.0 * (double)s * (double)s)) / (2.0 * (double)s);
            }

            s = sinMaxEndAngle / cosMaxEndAngle;

            signTest = (cosStartAngle > 0 && cosMaxEndAngle > 0) || (cosStartAngle < 0 && cosMaxEndAngle < 0);

            if (signTest) {
                maxN = (-((double)s * (double)t) - 1.0 + sqrt(((double)s * (double)t - 1.0) * ((double)s * (double)t - 1.0) + 4.0 * (double)s * (double)s)) / (2.0 * (double)s);
            }
            else {
                maxN = (-((double)s * (double)t) - 1.0 - sqrt(((double)s * (double)t - 1.0) * ((double)s * (double)t - 1.0) + 4.0 * (double)s * (double)s)) / (2.0 * (double)s);
            }
        }

        double minN1 = 32.0 * minN / 0.05;
        double maxN1 = 32.0 * maxN / 0.05;

        if (minN1 > maxN1) {
            double temp = minN1;
            minN1 = maxN1;
            maxN1 = temp;
        }

        minN1 = fmax(minN1, -32.0);
        maxN1 = fmin(maxN1, 32.0);

        if (minN1 <= maxN1) {
            double minMag = INFINITY;
            double maxMag = -INFINITY;

            double minYaw = INFINITY;
            double maxYaw = -INFINITY;

            double refYaw = 65536.0 * (atan2(minN1, minM1) / (2.0 * M_PI));
            refYaw = fmod(65536.0 + refYaw, 65536.0);

            for (int i = 0; i < 4; i++) {
                double m1;
                double n1;

                if (i % 2 == 0) {
                    m1 = minM1;
                }
                else {
                    m1 = maxM1;
                }

                if (i / 2 == 0) {
                    n1 = minN1;
                }
                else {
                    n1 = maxN1;
                }

                double mag = sqrt(m1 * m1 + n1 * n1);
                double yaw = 65536.0 * (atan2(n1, m1) / (2.0 * M_PI));
                yaw = fmod(65536.0 + yaw, 65536.0);

                double yawDiff = yaw - refYaw;
                yawDiff = fmod(fmod(yawDiff, 65536.0) + 98304.0, 65536.0) - 32768.0f;

                minMag = fmin(minMag, mag);
                maxMag = fmax(maxMag, mag);
                minYaw = fmin(minYaw, yawDiff);
                maxYaw = fmax(maxYaw, yawDiff);
            }

            maxMag = fmin(maxMag, 32.0);

            if (minMag <= maxMag) {
                int minIntendedDYaw = 16 * (int)ceil((minYaw + refYaw) / 16);
                int maxIntendedDYaw = 16 * (int)floor((maxYaw + refYaw) / 16);

                int minIdx = -1;
                int maxIdx = magCount;

                while (maxIdx > minIdx + 1) {
                    int midIdx = (maxIdx + minIdx) / 2;

                    if (minMag - 0.001 < magSet[midIdx]) {
                        maxIdx = midIdx;
                    }
                    else {
                        minIdx = midIdx;
                    }
                }

                int startMagIdx = maxIdx;

                minIdx = -1;
                maxIdx = magCount;

                while (maxIdx > minIdx + 1) {
                    int midIdx = (maxIdx + minIdx) / 2;

                    if (maxMag + 0.001 < magSet[midIdx]) {
                        maxIdx = midIdx;
                    }
                    else {
                        minIdx = midIdx;
                    }
                }

                int endMagIdx = minIdx;
                
                for (int intendedDYaw = minIntendedDYaw; intendedDYaw <= maxIntendedDYaw; intendedDYaw+=slideAngleSampleRate) {
                    for (int magIdx = startMagIdx; magIdx <= endMagIdx; magIdx++) {
                        float intendedMag = magSet[magIdx];
                        try_upwarp_slide(solIdx, angle, intendedDYaw, intendedMag);
                    }
                }
                
            }
        }
    }
}

__global__ void test_slide_angle() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int angleIdx = idx % maxAngleRange;
    idx = idx / maxAngleRange;

    if (idx < min(n10KSolutions, MAX_10K_SOLUTIONS)) {
        struct TenKSolution* tenKSol = &(tenKSolutions[idx]);
        int angle = tenKSol->minStartAngle + angleIdx * slideAngleSampleRate;

        if (angle <= tenKSol->maxStartAngle) {
            angle = angle % 65536;

            try_pu_slide_angle(idx, angle, tenKSol->minEndAngle, tenKSol->maxEndAngle, tenKSol->minM1, tenKSol->maxM1);
        }
    }
}

__global__ void find_slide_solutions() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(n10KSolutions, MAX_10K_SOLUTIONS)) {
        struct TenKSolution* tenKSol = &(tenKSolutions[idx]);
        struct SpeedSolution* speedSol = &(speedSolutions[tenKSol->speedSolutionIdx]);
        struct SKUpwarpSolution* skuwSol = &(skuwSolutions[speedSol->skuwSolutionIdx]);
        struct UpwarpSolution* uwSol = &(upwarpSolutions[skuwSol->uwIdx]);
        struct PlatformSolution* platSol = &(platSolutions[uwSol->platformSolutionIdx]);

        int maxTurnAngle = 522;

        double minDist = INFINITY;
        double maxDist = -INFINITY;

        double baseAngle = 0.0;

        double minEndAngle = 0.0;
        double maxEndAngle = 0.0;

        for (int i = 0; i < 3; i++) {
            double xDist = 65536.0 * uwSol->pux + platSol->endTriangles[platSol->endFloorIdx][i][0] - platSol->endPosition[0];
            double zDist = 65536.0 * uwSol->puz + platSol->endTriangles[platSol->endFloorIdx][i][2] - platSol->endPosition[2];

            double dist = sqrt(xDist * xDist + zDist * zDist);

            double angle = atan2(-xDist, -zDist);
            angle = fmod(2.0 * M_PI + angle, 2.0 * M_PI);

            if (i == 0) {
                baseAngle = angle;
            }
            else {
                minEndAngle = fmin(minEndAngle, angle - baseAngle);
                maxEndAngle = fmax(maxEndAngle, angle - baseAngle);
            }

            minDist = fmin(minDist, dist);
            maxDist = fmax(maxDist, dist);
        }

        double minSpeed = -minDist / (double)platSol->endTriangleNormals[platSol->endFloorIdx][1];
        double maxSpeed = -maxDist / (double)platSol->endTriangleNormals[platSol->endFloorIdx][1];

        double minM = minSpeed / (double)speedSol->returnSpeed;
        double maxM = maxSpeed / (double)speedSol->returnSpeed;

        double minM1 = 32.0 * ((minM - 0.92) / 0.02);
        double maxM1 = 32.0 * ((maxM - 0.92) / 0.02);

        if (minM1 > maxM1) {
            double temp = minM1;
            minM1 = maxM1;
            maxM1 = temp;
        }

        minM1 = fmax(minM1, -32.0);
        maxM1 = fmin(maxM1, 32.0);

        if (minM1 <= maxM1) {
            minEndAngle = minEndAngle + baseAngle;
            maxEndAngle = maxEndAngle + baseAngle;

            minEndAngle = 65536.0 * minEndAngle / (2.0 * M_PI);
            maxEndAngle = 65536.0 * maxEndAngle / (2.0 * M_PI);

            int minStartAngle = (int)ceil(minEndAngle) - maxTurnAngle;
            int maxStartAngle = (int)floor(maxEndAngle) + maxTurnAngle;

            minStartAngle = minStartAngle + 15;
            minStartAngle = minStartAngle - (minStartAngle % 16);

            tenKSol->minStartAngle = minStartAngle;
            tenKSol->maxStartAngle = maxStartAngle;
            tenKSol->minEndAngle = minEndAngle;
            tenKSol->maxEndAngle = maxEndAngle;
            tenKSol->maxM1 = maxM1;
            tenKSol->minM1 = minM1;

            int angleRange = ((maxStartAngle - minStartAngle)/slideAngleSampleRate) + 1;
            atomicMax(&maxAngleRange, angleRange);
        }
    }
}

__device__ bool try_pu_xz(float* normal, float* position, short(&current_triangles)[2][3][3], float(&triangle_normals)[2][3], double x, double z, int platSolIdx) {
    // For current (x, z) PU position, find range of yaws that
    // allow you to reach the PU platform from the original universe.

    float test_normal[3] = { normal[0], normal[1], normal[2] };
    float mario_pos[3] = { x + position[0], position[1], z + position[2] };

    short triangles[2][3][3];
    float normals[2][3];
    float mat[4][4];

    platform_logic(test_normal, mario_pos, triangles, normals, mat);

    if (check_inbounds(mario_pos)) {
        for (int i = 0; i < n_y_ranges; i++) {
            if (mario_pos[1] >= *(lower_y + i) && mario_pos[1] < *(upper_y + i)) {
                bool good_solution = false;

                for (int f = 0; f < n_floor_ranges; f++) {
                    if (mario_pos[1] >= lower_floor[f] && mario_pos[1] <= upper_floor[f]) {
                        good_solution = true;
                        break;
                    }
                }

                if (!good_solution) {
                    float floor_dist = 65536.0;
                    float speed = 65536.0 * sqrtf(x * x + z * z);

                    for (int f = 0; f < n_floor_ranges; f++) {
                        float f_dist = mario_pos[1] - lower_floor[f];

                        if (f_dist > 0) {
                            floor_dist = f_dist;
                        }
                        else {
                            break;
                        }
                    }

                    int falling_frames = (int)ceil((sqrt(2.0 * floor_dist + 1.0) + 1.0) / 2.0);

                    int closest_pu_dist = fmin(fmin(mario_pos[0] + pow(2, 31), pow(2, 31) - 1.0 - mario_pos[0]), fmin(mario_pos[2] + pow(2, 31), pow(2, 31) - 1.0 - mario_pos[2]));

                    if (closest_pu_dist >= speed / 4.0f) {
                        int total_falling_frames = (int)floor((pow(2, 32) - closest_pu_dist - 3.0 * speed / 2.0) / speed);

                        if (falling_frames <= total_falling_frames) {
                            good_solution = true;
                        }
                    }
                }

                if (good_solution) {
                    int solIdx = atomicAdd(&nUpwarpSolutions, 1);

                    if (solIdx < MAX_UPWARP_SOLUTIONS) {
                        UpwarpSolution solution;
                        solution.platformSolutionIdx = platSolIdx;
                        solution.pux = (int)roundf(x / 65536.0f);
                        solution.puz = (int)roundf(z / 65536.0f);
                        upwarpSolutions[solIdx] = solution;
                    }

                    break;
                }
            }
        }
    }

    return true;
}

__device__ bool try_pu_x(float* normal, float* position, short(&current_triangles)[2][3][3], float(&triangle_normals)[2][3], float(&T_start)[4][4], float(&T_tilt)[4][4], double x, double x1_min, double x1_max, double x2_min, double x2_max, double platform_min_x, double platform_max_x, double platform_min_z, double platform_max_z, double m, double c_min, double c_max, int q_steps, double max_speed, int platSolIdx) {
    double pu_platform_min_x = x + platform_min_x;
    double pu_platform_max_x = x + platform_max_x;

    double pu_gap = 65536.0 * q_steps;

    // Find maximal range of PUs along z axis from current x PU position
    double min_z_pu_idx = (m * pu_platform_min_x + c_min) / pu_gap;
    double max_z_pu_idx = (m * pu_platform_max_x + c_max) / pu_gap;

    if (min_z_pu_idx > max_z_pu_idx) {
        double temp = min_z_pu_idx;
        min_z_pu_idx = max_z_pu_idx;
        max_z_pu_idx = temp;
    }

    // Check max_x_pu_idx and min_x_pu_idx are in range for valid platform tilt.
    // Correct them if they're not.
    //
    // Possible for only part of the platform to be in range.
    // In this case just skip whole PU to avoid headaches later on.

    if (pu_platform_max_x > fmin(x1_min, x1_max) && pu_platform_min_x < fmax(x1_min, x1_max)) {
        double z1_min = m * x1_min + c_min;
        double z1_max = m * x1_max + c_max;
        double tilt_cutoff_z = (z1_max - z1_min) * (x - x1_min) / (x1_max - x1_min) + z1_min;

        if (x1_min > 0) {
            // Find new lower bound for z_pu
            double tilt_cutoff_pu_idx = (tilt_cutoff_z - platform_max_z) / pu_gap;
            min_z_pu_idx = fmax(min_z_pu_idx, tilt_cutoff_pu_idx);
        }
        else {
            // Find new upper bound for z_pu
            double tilt_cutoff_pu_idx = (tilt_cutoff_z - platform_min_z) / pu_gap;
            max_z_pu_idx = fmin(max_z_pu_idx, tilt_cutoff_pu_idx);
        }
    }

    if (pu_platform_max_x > fmin(x2_min, x2_max) && pu_platform_min_x < fmax(x2_min, x2_max)) {
        double z2_min = m * x2_min + c_min;
        double z2_max = m * x2_max + c_max;
        double tilt_cutoff_z = (z2_max - z2_min) * (x - x2_min) / (x2_max - x2_min) + z2_min;

        if (x2_min > 0) {
            // Find new upper bound for z_pu
            double tilt_cutoff_pu_idx = (tilt_cutoff_z - platform_min_z) / pu_gap;
            max_z_pu_idx = fmin(max_z_pu_idx, tilt_cutoff_pu_idx);
        }
        else {
            // Find new lower bound for z_pu
            double tilt_cutoff_pu_idx = (tilt_cutoff_z - platform_max_z) / pu_gap;
            min_z_pu_idx = fmax(min_z_pu_idx, tilt_cutoff_pu_idx);
        }
    }

    min_z_pu_idx = q_steps * ceil(min_z_pu_idx);
    max_z_pu_idx = q_steps * floor(max_z_pu_idx);

    double min_z_pu = 65536.0 * min_z_pu_idx;
    double max_z_pu = 65536.0 * max_z_pu_idx;

    double closest_z_pu_platform;

    if (min_z_pu < 0) {
        if (max_z_pu < 0) {
            closest_z_pu_platform = max_z_pu + platform_max_z - platform_min_z;
        }
        else {
            if (abs(min_z_pu) < abs(max_z_pu)) {
                closest_z_pu_platform = min_z_pu + platform_max_z - platform_min_z;
            }
            else {
                closest_z_pu_platform = max_z_pu + platform_min_z - platform_max_z;
            }
        }
    }
    else {
        closest_z_pu_platform = min_z_pu + platform_min_z - platform_max_z;
    }

    // Find the minimum speed to reach a valid PU from current x position.
    // If this exceeds our maximum allowed speed, then we can stop searching polygon
    // in this direction.
    double min_needed_speed = (4.0 / (double)q_steps) * sqrt((x + platform_max_z - platform_min_z) * (x + platform_max_z - platform_min_z) + (closest_z_pu_platform * closest_z_pu_platform)) / fmax(triangle_normals[0][1], triangle_normals[1][1]);

    if (min_needed_speed > max_speed) {
        return false;
    }
    else {
        double min_pu_oob_z;

        if (q_steps < 4) {
            // If we're terminating Mario's movement early, then we need to be sure that 
            // there's enough of a difference between the y normals of the platform's two 
            // triangles to force Mario into out of bounds

            double closest_oob = 9743.23; // An estimate, based on the platforms pivot

            double min_dist_oob = closest_oob / (fmax(triangle_normals[0][1], triangle_normals[1][1]) / fmin(triangle_normals[0][1], triangle_normals[1][1]) - 1.0);
            double min_dist_oob_z = sqrt(min_dist_oob * min_dist_oob - x * x);

            min_pu_oob_z = ceil(min_dist_oob_z / 262144.0) * pu_gap;
        }
        else {
            min_pu_oob_z = 0.0;
        }

        double T_diff00 = T_tilt[0][0] - T_start[0][0];
        double T_diff20 = T_tilt[2][0] - T_start[2][0];
        double T_diff02 = T_tilt[0][2] - T_start[0][2];
        double T_diff22 = T_tilt[2][2] - T_start[2][2];

        // Tolerance for picking PUs that may result 
        // in out of bounds displacements.
        //
        // If we're more than the dimensions of the platform 
        // away from being in-bounds then we probably can't
        // get an in-bounds displacement anyway.
        double disp_leeway = abs(platform_min_x - platform_max_x) + abs(platform_min_z - platform_max_z);

        // Search backwards from z=0
        for (double z = fmin(fmin(0.0, max_z_pu), -min_pu_oob_z); z + 8192 > min_z_pu; z -= pu_gap) {
            double base_platform_displacement_x = x * T_diff00 + z * T_diff20;
            double base_platform_displacement_z = x * T_diff02 + z * T_diff22;

            double bpd_x_mod = static_cast<int16_t>(static_cast<int>(x + platform_pos[0] + base_platform_displacement_x));
            double bpd_z_mod = static_cast<int16_t>(static_cast<int>(z + platform_pos[2] + base_platform_displacement_z));

            // Check if our likely horizontal platform displacement puts us out of bounds.
            // If so, skip checking this PU.
            if (abs(bpd_x_mod) < 8192 + disp_leeway && abs(bpd_z_mod) < 8192 + disp_leeway) {
                if (!try_pu_xz(normal, position, current_triangles, triangle_normals, x, z, platSolIdx)) {
                    break;
                }
            }
        }

        // Search forwards from z>0
        for (double z = fmax(fmax(q_steps * pu_gap, min_z_pu), min_pu_oob_z); z - 8192 < max_z_pu; z += pu_gap) {
            double base_platform_displacement_x = x * T_diff00 + z * T_diff20;
            double base_platform_displacement_z = x * T_diff02 + z * T_diff22;

            double bpd_x_mod = static_cast<int16_t>(static_cast<int>(x + platform_pos[0] + base_platform_displacement_x));
            double bpd_z_mod = static_cast<int16_t>(static_cast<int>(z + platform_pos[2] + base_platform_displacement_z));

            // Check if our likely horizontal platform displacement puts us out of bounds.
            // If so, skip checking this PU.
            if (abs(bpd_x_mod) < 8192 + disp_leeway && abs(bpd_z_mod) < 8192 + disp_leeway) {
                if (!try_pu_xz(normal, position, current_triangles, triangle_normals, x, z, platSolIdx)) {
                    break;
                }
            }
        }

        return true;
    }
}

__device__ bool try_pu_z(float* normal, float* position, short(&current_triangles)[2][3][3], float(&triangle_normals)[2][3], float(&T_start)[4][4], float(&T_tilt)[4][4], double z, double z1_min, double z1_max, double z2_min, double z2_max, double platform_min_x, double platform_max_x, double platform_min_z, double platform_max_z, double m, double c_min, double c_max, int q_steps, double max_speed, int platSolIdx) {
    double pu_platform_min_z = z + platform_min_z;
    double pu_platform_max_z = z + platform_max_z;

    double pu_gap = 65535.0 * q_steps;

    // Find maximal range of PUs along x axis from current z PU position
    double min_x_pu_idx = ((pu_platform_min_z - c_min) / m) / pu_gap;
    double max_x_pu_idx = ((pu_platform_max_z - c_max) / m) / pu_gap;

    if (min_x_pu_idx > max_x_pu_idx) {
        double temp = min_x_pu_idx;
        min_x_pu_idx = max_x_pu_idx;
        max_x_pu_idx = temp;
    }

    // Check max_x_pu and min_x_pu are in range for valid platform tilt.
    // Correct them if they're not.
    //
    // Possible for only part of the platform to be in range.
    // In this case just skip it to avoid headaches later on.

    if (pu_platform_max_z > fmin(z1_min, z1_max) && pu_platform_min_z < fmax(z1_min, z1_max)) {
        double x1_min = (z1_min - c_min) / m;
        double x1_max = (z1_max - c_max) / m;
        double tilt_cutoff_x = (x1_max - x1_min) * (z - z1_min) / (z1_max - z1_min) + x1_min;

        if (z1_min > 0) {
            // Find new upper bound for z_pu
            double tilt_cutoff_pu_idx = (tilt_cutoff_x - platform_min_x) / pu_gap;
            max_x_pu_idx = fmin(max_x_pu_idx, tilt_cutoff_pu_idx);
        }
        else {
            // Find new lower bound for z_pu
            double tilt_cutoff_pu_idx = (tilt_cutoff_x - platform_max_x) / pu_gap;
            min_x_pu_idx = fmax(min_x_pu_idx, tilt_cutoff_pu_idx);
        }
    }

    if (pu_platform_max_z > fmin(z2_min, z2_max) && pu_platform_min_z < fmax(z2_min, z2_max)) {
        double x2_min = (z2_min - c_min) / m;
        double x2_max = (z2_max - c_max) / m;
        double tilt_cutoff_x = (x2_max - x2_min) * (z - z2_min) / (z2_max - z2_min) + x2_min;

        if (z2_min > 0) {
            // Find new lower bound for z_pu
            double tilt_cutoff_pu_idx = (tilt_cutoff_x - platform_max_x) / pu_gap;
            min_x_pu_idx = fmax(min_x_pu_idx, tilt_cutoff_pu_idx);
        }
        else {
            // Find new upper bound for z_pu
            double tilt_cutoff_pu_idx = (tilt_cutoff_x - platform_min_x) / pu_gap;
            max_x_pu_idx = fmin(max_x_pu_idx, tilt_cutoff_pu_idx);
        }
    }

    min_x_pu_idx = q_steps * ceil(min_x_pu_idx);
    max_x_pu_idx = q_steps * floor(max_x_pu_idx);

    double min_x_pu = 65536.0 * min_x_pu_idx;
    double max_x_pu = 65536.0 * max_x_pu_idx;

    double closest_x_pu_platform;

    if (min_x_pu < 0) {
        if (max_x_pu < 0) {
            closest_x_pu_platform = max_x_pu + platform_max_x - platform_min_x;
        }
        else {
            if (abs(min_x_pu) < abs(max_x_pu)) {
                closest_x_pu_platform = min_x_pu + platform_max_x - platform_min_x;
            }
            else {
                closest_x_pu_platform = max_x_pu + platform_min_x - platform_max_x;
            }
        }
    }
    else {
        closest_x_pu_platform = min_x_pu + platform_min_x - platform_max_x;
    }

    // Find the minimum speed to reach a valid PU from current z position.
    // If this exceeds our maximum allowed speed, then we can stop searching
    // the polygon in this direction.
    double min_needed_speed = (4.0 / (double)q_steps) * sqrt((z + platform_max_x - platform_min_x) * (z + platform_max_x - platform_min_x) + (closest_x_pu_platform * closest_x_pu_platform)) / fmax(triangle_normals[0][1], triangle_normals[1][1]);

    if (min_needed_speed > max_speed) {
        return false;
    }
    else {
        double min_pu_oob_x;

        if (q_steps < 4) {
            // If we're terminating Mario's movement early, then we need to be sure that 
            // there's enough of a difference between the y normals of the platform's two 
            // triangles to force Mario into out of bounds

            double closest_oob = 9743.23; // An estimate, based on the platform's pivot

            double min_dist_oob = closest_oob / (fmax(triangle_normals[0][1], triangle_normals[1][1]) / fmin(triangle_normals[0][1], triangle_normals[1][1]) - 1.0);
            double min_dist_oob_x = sqrt(min_dist_oob * min_dist_oob - z * z);

            min_pu_oob_x = ceil(min_dist_oob_x / 262144.0) * pu_gap;
        }
        else {
            min_pu_oob_x = 0.0;
        }

        double T_diff00 = T_tilt[0][0] - T_start[0][0];
        double T_diff20 = T_tilt[2][0] - T_start[2][0];
        double T_diff02 = T_tilt[0][2] - T_start[0][2];
        double T_diff22 = T_tilt[2][2] - T_start[2][2];

        // Tolerance for picking PUs that may result 
        // in out of bounds displacements.
        //
        // If we're more than the dimensions of the platform 
        // away from being in-bounds then we probably can't
        // get an in-bounds displacement anyway.
        double disp_leeway = abs(platform_min_x - platform_max_x) + abs(platform_min_z - platform_max_z);

        // Search backwards from x=0
        for (double x = fmin(fmin(0.0, max_x_pu), -min_pu_oob_x); x + 8192 > min_x_pu; x -= pu_gap) {
            double base_platform_displacement_x = x * T_diff00 + z * T_diff20;
            double base_platform_displacement_z = x * T_diff02 + z * T_diff22;

            double bpd_x_mod = static_cast<int16_t>(static_cast<int>(x + platform_pos[0] + base_platform_displacement_x));
            double bpd_z_mod = static_cast<int16_t>(static_cast<int>(z + platform_pos[2] + base_platform_displacement_z));

            // Check if our likely horizontal platform displacement puts us out of bounds.
            // If so, skip checking this PU.
            if (abs(bpd_x_mod) < 8192 + disp_leeway && abs(bpd_z_mod) < 8192 + disp_leeway) {
                if (!try_pu_xz(normal, position, current_triangles, triangle_normals, x, z, platSolIdx)) {
                    break;
                }
            }
        }

        // Search forwards from x>0
        for (double x = fmax(fmax(pu_gap, min_x_pu), min_pu_oob_x); x - 8192 < max_x_pu; x += pu_gap) {
            double base_platform_displacement_x = x * T_diff00 + z * T_diff20;
            double base_platform_displacement_z = x * T_diff02 + z * T_diff22;

            double bpd_x_mod = static_cast<int16_t>(static_cast<int>(x + platform_pos[0] + base_platform_displacement_x));
            double bpd_z_mod = static_cast<int16_t>(static_cast<int>(z + platform_pos[2] + base_platform_displacement_z));

            // Check if our likely horizontal platform displacement puts us out of bounds.
            // If so, skip checking this PU.
            if (abs(bpd_x_mod) < 8192 + disp_leeway && abs(bpd_z_mod) < 8192 + disp_leeway) {
                if (!try_pu_xz(normal, position, current_triangles, triangle_normals, x, z, platSolIdx)) {
                    break;
                }
            }
        }

        return true;
    }
}

__device__ void try_normal(float* normal, float* position, int platSolIdx, double max_speed) {
    // Tilt angle cut-offs
    // These are the yaw boundaries where the platform tilt 
    // switches direction. Directions match normal_offsets:
    // Between a[0] and a[1]: +x +z
    // Between a[1] and a[2]: -x +z
    // Between a[2] and a[3]: -x -z
    // Between a[3] and a[0]: +x -z

    short current_triangles[2][3][3];
    float triangle_normals[2][3];

    float T_start[4][4];
    T_start[1][0] = normal[0];
    T_start[1][1] = normal[1];
    T_start[1][2] = normal[2];

    float invsqrt = 1.0f / sqrtf(T_start[1][0] * T_start[1][0] + T_start[1][1] * T_start[1][1] + T_start[1][2] * T_start[1][2]);

    T_start[1][0] *= invsqrt;
    T_start[1][1] *= invsqrt;
    T_start[1][2] *= invsqrt;

    T_start[0][0] = T_start[1][1] * 1.0f - 0.0f * T_start[1][2];
    T_start[0][1] = T_start[1][2] * 0.0f - 1.0f * T_start[1][0];
    T_start[0][2] = T_start[1][0] * 0.0f - 0.0f * T_start[1][1];

    invsqrt = 1.0f / sqrtf(T_start[0][0] * T_start[0][0] + T_start[0][1] * T_start[0][1] + T_start[0][2] * T_start[0][2]);

    T_start[0][0] *= invsqrt;
    T_start[0][1] *= invsqrt;
    T_start[0][2] *= invsqrt;

    T_start[2][0] = T_start[0][1] * T_start[1][2] - T_start[1][1] * T_start[0][2];
    T_start[2][1] = T_start[0][2] * T_start[1][0] - T_start[1][2] * T_start[0][0];
    T_start[2][2] = T_start[0][0] * T_start[1][1] - T_start[1][0] * T_start[0][1];

    invsqrt = 1.0f / sqrtf(T_start[2][0] * T_start[2][0] + T_start[2][1] * T_start[2][1] + T_start[2][2] * T_start[2][2]);

    T_start[2][0] *= invsqrt;
    T_start[2][1] *= invsqrt;
    T_start[2][2] *= invsqrt;

    T_start[3][0] = platform_pos[0];
    T_start[3][1] = platform_pos[1];
    T_start[3][2] = platform_pos[2];
    T_start[0][3] = 0.0f;
    T_start[1][3] = 0.0f;
    T_start[2][3] = 0.0f;
    T_start[3][3] = 1.0f;

    for (int h = 0; h < 2; h++) {
        for (int i = 0; i < 3; i++) {
            float vx = default_triangles[h][i][0];
            float vy = default_triangles[h][i][1];
            float vz = default_triangles[h][i][2];

            current_triangles[h][i][0] = (short)(int)(vx * T_start[0][0] + vy * T_start[1][0] + vz * T_start[2][0] + T_start[3][0]);
            current_triangles[h][i][1] = (short)(int)(vx * T_start[0][1] + vy * T_start[1][1] + vz * T_start[2][1] + T_start[3][1]);
            current_triangles[h][i][2] = (short)(int)(vx * T_start[0][2] + vy * T_start[1][2] + vz * T_start[2][2] + T_start[3][2]);
        }

        triangle_normals[h][0] = ((current_triangles[h][1][1] - current_triangles[h][0][1]) * (current_triangles[h][2][2] - current_triangles[h][1][2])) - ((current_triangles[h][1][2] - current_triangles[h][0][2]) * (current_triangles[h][2][1] - current_triangles[h][1][1]));
        triangle_normals[h][1] = ((current_triangles[h][1][2] - current_triangles[h][0][2]) * (current_triangles[h][2][0] - current_triangles[h][1][0])) - ((current_triangles[h][1][0] - current_triangles[h][0][0]) * (current_triangles[h][2][2] - current_triangles[h][1][2]));
        triangle_normals[h][2] = ((current_triangles[h][1][0] - current_triangles[h][0][0]) * (current_triangles[h][2][1] - current_triangles[h][1][1])) - ((current_triangles[h][1][1] - current_triangles[h][0][1]) * (current_triangles[h][2][0] - current_triangles[h][1][0]));

        invsqrt = 1.0f / sqrtf(triangle_normals[h][0] * triangle_normals[h][0] + triangle_normals[h][1] * triangle_normals[h][1] + triangle_normals[h][2] * triangle_normals[h][2]);

        triangle_normals[h][0] *= invsqrt;
        triangle_normals[h][1] *= invsqrt;
        triangle_normals[h][2] *= invsqrt;
    }

    float nx = normal[0];
    float ny = normal[1];
    float nz = normal[2];

    double a[4];
    a[0] = atan2(nz, sqrt(1 - nz * nz));
    a[1] = atan2(sqrt(1 - nx * nx), nx);
    a[2] = M_PI - a[0];
    a[3] = 2 * M_PI - a[1];

    double platform_min_x = fmin(fmin((double)current_triangles[0][0][0], (double)current_triangles[0][1][0]), fmin((double)current_triangles[0][2][0], (double)current_triangles[1][2][0]));
    double platform_max_x = fmax(fmax((double)current_triangles[0][0][0], (double)current_triangles[0][1][0]), fmax((double)current_triangles[0][2][0], (double)current_triangles[1][2][0]));
    double platform_min_z = fmin(fmin((double)current_triangles[0][0][2], (double)current_triangles[0][1][2]), fmin((double)current_triangles[0][2][2], (double)current_triangles[1][2][2]));
    double platform_max_z = fmax(fmax((double)current_triangles[0][0][2], (double)current_triangles[0][1][2]), fmax((double)current_triangles[0][2][2], (double)current_triangles[1][2][2]));

    double min_y = fmin(-3071.0, fmin(fmin((double)current_triangles[0][0][1], (double)current_triangles[0][1][1]), fmin((double)current_triangles[0][2][1], (double)current_triangles[1][2][1])));
    double max_y = fmax(fmax((double)current_triangles[0][0][1], (double)current_triangles[0][1][1]), fmax((double)current_triangles[0][2][1], (double)current_triangles[1][2][1]));

    // Try to find solutions for each possible platform tilt direction
    for (int i = 0; i < 4; i++) {
        float T_tilt[4][4];
        T_tilt[1][0] = normal[0] + normal_offsets[i][0];
        T_tilt[1][1] = normal[1] + normal_offsets[i][1];
        T_tilt[1][2] = normal[2] + normal_offsets[i][2];

        float invsqrt = 1.0f / sqrtf(T_tilt[1][0] * T_tilt[1][0] + T_tilt[1][1] * T_tilt[1][1] + T_tilt[1][2] * T_tilt[1][2]);

        T_tilt[1][0] *= invsqrt;
        T_tilt[1][1] *= invsqrt;
        T_tilt[1][2] *= invsqrt;

        T_tilt[0][0] = T_tilt[1][1] * 1.0f - 0.0f * T_tilt[1][2];
        T_tilt[0][1] = T_tilt[1][2] * 0.0f - 1.0f * T_tilt[1][0];
        T_tilt[0][2] = T_tilt[1][0] * 0.0f - 0.0f * T_tilt[1][1];

        invsqrt = 1.0f / sqrtf(T_tilt[0][0] * T_tilt[0][0] + T_tilt[0][1] * T_tilt[0][1] + T_tilt[0][2] * T_tilt[0][2]);

        T_tilt[0][0] *= invsqrt;
        T_tilt[0][1] *= invsqrt;
        T_tilt[0][2] *= invsqrt;

        T_tilt[2][0] = T_tilt[0][1] * T_tilt[1][2] - T_tilt[1][1] * T_tilt[0][2];
        T_tilt[2][1] = T_tilt[0][2] * T_tilt[1][0] - T_tilt[1][2] * T_tilt[0][0];
        T_tilt[2][2] = T_tilt[0][0] * T_tilt[1][1] - T_tilt[1][0] * T_tilt[0][1];

        invsqrt = 1.0f / sqrtf(T_tilt[2][0] * T_tilt[2][0] + T_tilt[2][1] * T_tilt[2][1] + T_tilt[2][2] * T_tilt[2][2]);

        T_tilt[2][0] *= invsqrt;
        T_tilt[2][1] *= invsqrt;
        T_tilt[2][2] *= invsqrt;

        T_tilt[3][0] = platform_pos[0];
        T_tilt[3][1] = platform_pos[1];
        T_tilt[3][2] = platform_pos[2];
        T_tilt[0][3] = 0.0f;
        T_tilt[1][3] = 0.0f;
        T_tilt[2][3] = 0.0f;
        T_tilt[3][3] = 1.0f;

        double T_diff01 = T_tilt[0][1] - T_start[0][1];
        double T_diff11 = T_tilt[1][1] - T_start[1][1];
        double T_diff21 = T_tilt[2][1] - T_start[2][1];

        for (int j = 0; j < n_y_ranges; j++) {
            double r_min = lower_y[j] - (1 + T_diff11) * max_y + T_diff01 * platform_pos[0] + T_diff11 * platform_pos[1] + T_diff21 * platform_pos[2];
            double r_max = upper_y[j] - (1 + T_diff11) * min_y + T_diff01 * platform_pos[0] + T_diff11 * platform_pos[1] + T_diff21 * platform_pos[2];

            // z = mx + c_min
            // z = mx + c_max
            //
            // PU platforms occurring between these lines will (usually) 
            // give a y displacement within our desired range.
            double m = -T_diff01 / T_diff21;
            double c_min; double c_max;

            if (T_diff21 < 0) {
                c_min = r_max / T_diff21;
                c_max = r_min / T_diff21;
            }
            else {
                c_min = r_min / T_diff21;
                c_max = r_max / T_diff21;
            }

            // Find intersection between y displacement lines and 
            // good platform tilt angle ranges.
            //
            // Intersection forms a polygon that may (or may not)
            // stretch to infinity in one direction.
            // 
            // Find the x coordinates where displacement lines and 
            // platform tilt lines intersect.
            //
            // Non-intersecting lines have x coordinate set to NaN. 
            double a1_cos = cos(a[i]);
            double a2_cos = cos(a[(i + 1) % 4]);

            double x1_min; double x1_max; double x2_min; double x2_max;

            if (nx == 0) {
                if (i % 2 == 0) {
                    x1_min = (c_min + tan(a[i]) * platform_pos[0] - platform_pos[2]) / (tan(a[i]) - m);
                    x1_max = (c_max + tan(a[i]) * platform_pos[0] - platform_pos[2]) / (tan(a[i]) - m);
                    x2_min = 0;
                    x2_max = 0;

                    if (a1_cos > 0 && x1_min < platform_pos[0] || a1_cos < 0 && x1_min > platform_pos[0]) {
                        x1_min = NAN;
                    }

                    if (a1_cos > 0 && x1_max < platform_pos[0] || a1_cos < 0 && x1_max > platform_pos[0]) {
                        x1_max = NAN;
                    }

                    if (nz > 0 && c_min < platform_pos[0] || nz < 0 && c_min > platform_pos[0]) {
                        x2_min = NAN;
                    }

                    if (nz > 0 && c_max < platform_pos[0] || nz < 0 && c_max > platform_pos[0]) {
                        x2_max = NAN;
                    }
                }
                else {
                    x1_min = 0;
                    x1_max = 0;
                    x2_min = (c_min + tan(a[(i + 1) % 4]) * platform_pos[0] - platform_pos[2]) / (tan(a[(i + 1) % 4]) - m);
                    x2_max = (c_max + tan(a[(i + 1) % 4]) * platform_pos[0] - platform_pos[2]) / (tan(a[(i + 1) % 4]) - m);

                    if (nz > 0 && c_min < platform_pos[0] || nz < 0 && c_min > platform_pos[0]) {
                        x1_min = NAN;
                    }

                    if (nz > 0 && c_max < platform_pos[0] || nz < 0 && c_max > platform_pos[0]) {
                        x1_max = NAN;
                    }

                    if (a2_cos > 0 && x2_min < platform_pos[0] || a2_cos < 0 && x2_min > platform_pos[0]) {
                        x2_min = NAN;
                    }

                    if (a2_cos > 0 && x2_max < platform_pos[0] || a2_cos < 0 && x2_max >platform_pos[0]) {
                        x2_max = NAN;
                    }
                }
            }
            else {
                x1_min = (c_min + tan(a[i]) * platform_pos[0] - platform_pos[2]) / (tan(a[i]) - m);
                x1_max = (c_max + tan(a[i]) * platform_pos[0] - platform_pos[2]) / (tan(a[i]) - m);
                x2_min = (c_min + tan(a[(i + 1) % 4]) * platform_pos[0] - platform_pos[2]) / (tan(a[(i + 1) % 4]) - m);
                x2_max = (c_max + tan(a[(i + 1) % 4]) * platform_pos[0] - platform_pos[2]) / (tan(a[(i + 1) % 4]) - m);

                if (a1_cos > 0 && x1_min < platform_pos[0] || a1_cos < 0 && x1_min > platform_pos[0]) {
                    x1_min = NAN;
                }

                if (a1_cos > 0 && x1_max < platform_pos[0] || a1_cos < 0 && x1_max > platform_pos[0]) {
                    x1_max = NAN;
                }

                if (a2_cos > 0 && x2_min < platform_pos[0] || a2_cos < 0 && x2_min > platform_pos[0]) {
                    x2_min = NAN;
                }

                if (a2_cos > 0 && x2_max < platform_pos[0] || a2_cos < 0 && x2_max > platform_pos[0]) {
                    x2_max = NAN;
                }
            }


            // Mario's movement can end on any of his quarter steps, as long as the next move puts him 
            // out of bounds (or is the last step). So we need to consider PU movement for each possible
            // final quarter step

            // If the normals match then you can't force Mario out of bounds after his final q step.
            // Therefore, only 4 q_steps are possible.
            int q = 4;
            double pu_gap = 65536.0 * q;

            // Start searching for PUs in the polygon.
            //
            // We want to minimise speed, so we search outwards
            // from the point closest to the real platform.
            //
            // This will be at the x = 0 (if abs(m) < 1)
            // or z = 0 (if abs(m) > 1)
            if (abs(m) < 1) {
                // Find x limits of polygon
                double poly_x_start; double poly_x_end;

                if (!isnan(x1_min) && !isnan(x1_max)) {
                    if (!isnan(x2_min) && !isnan(x2_max)) {
                        poly_x_start = fmin(fmin(x1_min, x1_max), fmin(x2_min, x2_max));
                        poly_x_end = fmax(fmax(x1_min, x1_max), fmax(x2_min, x2_max));
                    }
                    else {
                        if (c_min > 0) {
                            poly_x_start = -INFINITY;
                            poly_x_end = fmax(x1_min, x1_max);
                        }
                        else {
                            poly_x_start = fmin(x1_min, x1_max);
                            poly_x_end = INFINITY;
                        }
                    }
                }
                else if (!isnan(x2_min) && !isnan(x2_max)) {
                    if (c_min > 0) {
                        poly_x_start = fmin(x2_min, x2_max);
                        poly_x_end = INFINITY;
                    }
                    else {
                        poly_x_start = -INFINITY;
                        poly_x_end = fmax(x2_min, x2_max);
                    }
                }
                else {
                    continue;
                }

                double first_x_pu = ceil((poly_x_start - platform_max_x) / pu_gap) * pu_gap;
                double last_x_pu = floor((poly_x_end - platform_min_x) / pu_gap) * pu_gap;

                // Search backwards from x=0
                for (double x = fmin(0.0, last_x_pu); x + platform_min_x > poly_x_start; x -= pu_gap) {
                    if (!try_pu_x(normal, position, current_triangles, triangle_normals, T_start, T_tilt, x, x1_min, x1_max, x2_min, x2_max, platform_min_x, platform_max_x, platform_min_z, platform_max_z, m, c_min, c_max, q, max_speed, platSolIdx)) {
                        break;
                    }
                }

                // Search forwards from x>0
                for (double x = fmax(pu_gap, first_x_pu); x - platform_max_x < poly_x_end; x += pu_gap) {
                    if (!try_pu_x(normal, position, current_triangles, triangle_normals, T_start, T_tilt, x, x1_min, x1_max, x2_min, x2_max, platform_min_x, platform_max_x, platform_min_z, platform_max_z, m, c_min, c_max, q, max_speed, platSolIdx)) {
                        break;
                    }
                }
            }
            else {
                // Calculate z coordinates of intersection points
                double z1_min = tan(a[i]) * x1_min + platform_pos[2] - tan(a[i]) * platform_pos[0];
                double z1_max = tan(a[i]) * x1_max + platform_pos[2] - tan(a[i]) * platform_pos[0];
                double z2_min = tan(a[(i + 1) % 4]) * x2_min + platform_pos[2] - tan(a[(i + 1) % 4]) * platform_pos[0];
                double z2_max = tan(a[(i + 1) % 4]) * x2_max + platform_pos[2] - tan(a[(i + 1) % 4]) * platform_pos[0];

                // Find z limits of polygon
                double poly_z_start; double poly_z_end;

                if (!isnan(z1_min) && !isnan(z1_max)) {
                    if (!isnan(z2_min) && !isnan(z2_max)) {
                        poly_z_start = fmin(fmin(z1_min, z1_max), fmin(z2_min, z2_max));
                        poly_z_end = fmax(fmax(z1_min, z1_max), fmax(z2_min, z2_max));
                    }
                    else {
                        if (c_min / m > 0) {
                            poly_z_start = -INFINITY;
                            poly_z_end = fmax(z1_min, z1_max);
                        }
                        else {
                            poly_z_start = fmin(z1_min, z1_max);
                            poly_z_end = INFINITY;
                        }
                    }
                }
                else if (!isnan(z2_min) && !isnan(z2_max)) {
                    if (c_min / m > 0) {
                        poly_z_start = fmin(z2_min, z2_max);
                        poly_z_end = INFINITY;
                    }
                    else {
                        poly_z_start = -INFINITY;
                        poly_z_end = fmax(z2_min, z2_max);
                    }
                }
                else {
                    continue;
                }

                double first_z_pu = ceil((poly_z_start - platform_max_z) / pu_gap) * pu_gap;
                double last_z_pu = floor((poly_z_end - platform_min_z) / pu_gap) * pu_gap;

                // Search backwards from z=0
                for (double z = fmin(0.0, last_z_pu); z + platform_min_z > poly_z_start; z -= pu_gap) {
                    if (!try_pu_z(normal, position, current_triangles, triangle_normals, T_start, T_tilt, z, z1_min, z1_max, z2_min, z2_max, platform_min_x, platform_max_x, platform_min_z, platform_max_z, m, c_min, c_max, q, max_speed, platSolIdx)) {
                        break;
                    }
                }

                // Search forwards from z>0
                for (double z = fmax(pu_gap, first_z_pu); z - platform_max_z < poly_z_end; z += pu_gap) {
                    if (!try_pu_z(normal, position, current_triangles, triangle_normals, T_start, T_tilt, z, z1_min, z1_max, z2_min, z2_max, platform_min_x, platform_max_x, platform_min_z, platform_max_z, m, c_min, c_max, q, max_speed, platSolIdx)) {
                        break;
                    }
                }
            }
        }
    }
}

__global__ void find_upwarp_solutions(float maxSpeed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nPlatSolutions, MAX_PLAT_SOLUTIONS)) {
        struct PlatformSolution* platSol = &(platSolutions[idx]);
        try_normal(platSol->endNormal, platSol->endPosition, idx, maxSpeed);
    }
}

__device__ void try_position(float* marioPos, float* normal, int maxFrames) {
    const float platformPos[3] = { platform_pos[0], platform_pos[1], platform_pos[2] };
    const short defaultTriangles[2][3][3] = { {{307, 307, -306}, {-306, 307, -306}, {-306, 307, 307}}, {{307, 307, -306}, {-306, 307, 307}, {307, 307, 307}} };

    float mat[4][4];
    mat[1][0] = (normal[0] <= 0.0f) ? ((0.0f - normal[0] < 0.01f) ? 0.0f : (normal[0] + 0.01f)) : ((0.0f - normal[0] > -0.01f) ? 0.0f : (normal[0] - 0.01f));
    mat[1][1] = (normal[1] <= 1.0f) ? ((1.0f - normal[1] < 0.01f) ? 1.0f : (normal[1] + 0.01f)) : ((1.0f - normal[1] > -0.01f) ? 1.0f : (normal[1] - 0.01f));
    mat[1][2] = (normal[2] <= 0.0f) ? ((0.0f - normal[2] < 0.01f) ? 0.0f : (normal[2] + 0.01f)) : ((0.0f - normal[2] > -0.01f) ? 0.0f : (normal[2] - 0.01f));

    float invsqrt = 1.0f / sqrtf(mat[1][0] * mat[1][0] + mat[1][1] * mat[1][1] + mat[1][2] * mat[1][2]);

    mat[1][0] *= invsqrt;
    mat[1][1] *= invsqrt;
    mat[1][2] *= invsqrt;

    mat[0][0] = mat[1][1] * 1.0f - 0.0f * mat[1][2];
    mat[0][1] = mat[1][2] * 0.0f - 1.0f * mat[1][0];
    mat[0][2] = mat[1][0] * 0.0f - 0.0f * mat[1][1];

    invsqrt = 1.0f / sqrtf(mat[0][0] * mat[0][0] + mat[0][1] * mat[0][1] + mat[0][2] * mat[0][2]);

    mat[0][0] *= invsqrt;
    mat[0][1] *= invsqrt;
    mat[0][2] *= invsqrt;

    mat[2][0] = mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2];
    mat[2][1] = mat[0][2] * mat[1][0] - mat[1][2] * mat[0][0];
    mat[2][2] = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];

    invsqrt = 1.0f / sqrtf(mat[2][0] * mat[2][0] + mat[2][1] * mat[2][1] + mat[2][2] * mat[2][2]);

    mat[2][0] *= invsqrt;
    mat[2][1] *= invsqrt;
    mat[2][2] *= invsqrt;

    mat[3][0] = platformPos[0];
    mat[3][1] = platformPos[1];
    mat[3][2] = platformPos[2];
    mat[0][3] = 0.0f;
    mat[1][3] = 0.0f;
    mat[2][3] = 0.0f;
    mat[3][3] = 1.0f;

    short currentTriangles[2][3][3];
    float triangleNormals[2][3];

    for (int h = 0; h < 2; h++) {
        for (int i = 0; i < 3; i++) {
            float vx = defaultTriangles[h][i][0];
            float vy = defaultTriangles[h][i][1];
            float vz = defaultTriangles[h][i][2];

            currentTriangles[h][i][0] = (short)(int)(vx * mat[0][0] + vy * mat[1][0] + vz * mat[2][0] + mat[3][0]);
            currentTriangles[h][i][1] = (short)(int)(vx * mat[0][1] + vy * mat[1][1] + vz * mat[2][1] + mat[3][1]);
            currentTriangles[h][i][2] = (short)(int)(vx * mat[0][2] + vy * mat[1][2] + vz * mat[2][2] + mat[3][2]);
        }

        triangleNormals[h][0] = ((currentTriangles[h][1][1] - currentTriangles[h][0][1]) * (currentTriangles[h][2][2] - currentTriangles[h][1][2])) - ((currentTriangles[h][1][2] - currentTriangles[h][0][2]) * (currentTriangles[h][2][1] - currentTriangles[h][1][1]));
        triangleNormals[h][1] = ((currentTriangles[h][1][2] - currentTriangles[h][0][2]) * (currentTriangles[h][2][0] - currentTriangles[h][1][0])) - ((currentTriangles[h][1][0] - currentTriangles[h][0][0]) * (currentTriangles[h][2][2] - currentTriangles[h][1][2]));
        triangleNormals[h][2] = ((currentTriangles[h][1][0] - currentTriangles[h][0][0]) * (currentTriangles[h][2][1] - currentTriangles[h][1][1])) - ((currentTriangles[h][1][1] - currentTriangles[h][0][1]) * (currentTriangles[h][2][0] - currentTriangles[h][1][0]));

        invsqrt = 1.0f / sqrtf(triangleNormals[h][0] * triangleNormals[h][0] + triangleNormals[h][1] * triangleNormals[h][1] + triangleNormals[h][2] * triangleNormals[h][2]);

        triangleNormals[h][0] *= invsqrt;
        triangleNormals[h][1] *= invsqrt;
        triangleNormals[h][2] *= invsqrt;
    }

    float floor_height = 0.0;
    int floor_idx = -1;

    short x = (short)(int)marioPos[0];
    short y = (short)(int)marioPos[1];
    short z = (short)(int)marioPos[2];

    for (int i = 0; i < 2; i++) {
        short x1 = currentTriangles[i][0][0];
        short z1 = currentTriangles[i][0][2];
        short x2 = currentTriangles[i][1][0];
        short z2 = currentTriangles[i][1][2];

        // Check that the point is within the triangle bounds.
        if ((z1 - z) * (x2 - x1) - (x1 - x) * (z2 - z1) < 0) {
            continue;
        }

        // To slightly save on computation time, set this later.
        int16_t x3 = currentTriangles[i][2][0];
        int16_t z3 = currentTriangles[i][2][2];

        if ((z2 - z) * (x3 - x2) - (x2 - x) * (z3 - z2) < 0) {
            continue;
        }
        if ((z3 - z) * (x1 - x3) - (x3 - x) * (z1 - z3) < 0) {
            continue;
        }

        float nx = triangleNormals[i][0];
        float ny = triangleNormals[i][1];
        float nz = triangleNormals[i][2];
        float oo = -(nx * x1 + ny * currentTriangles[i][0][1] + nz * z1);

        // Find the height of the floor at a given location.
        float height = -(x * nx + nz * z + oo) / ny;
        // Checks for floor interaction with a 78 unit buffer.
        if (y - (height + -78.0f) < 0.0f) {
            continue;
        }

        floor_height = height;
        floor_idx = i;
        break;
    }

    if (floor_idx != -1 && floor_height - 100.0f > -3071.0f && floor_height >= -2967.168)
    {
        marioPos[1] = floor_height;

        mat[1][0] = normal[0];
        mat[1][1] = normal[1];
        mat[1][2] = normal[2];

        invsqrt = 1.0f / sqrtf(mat[1][0] * mat[1][0] + mat[1][1] * mat[1][1] + mat[1][2] * mat[1][2]);

        mat[1][0] *= invsqrt;
        mat[1][1] *= invsqrt;
        mat[1][2] *= invsqrt;

        mat[0][0] = mat[1][1] * 1.0f - 0.0f * mat[1][2];
        mat[0][1] = mat[1][2] * 0.0f - 1.0f * mat[1][0];
        mat[0][2] = mat[1][0] * 0.0f - 0.0f * mat[1][1];

        invsqrt = 1.0f / sqrtf(mat[0][0] * mat[0][0] + mat[0][1] * mat[0][1] + mat[0][2] * mat[0][2]);

        mat[0][0] *= invsqrt;
        mat[0][1] *= invsqrt;
        mat[0][2] *= invsqrt;

        mat[2][0] = mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2];
        mat[2][1] = mat[0][2] * mat[1][0] - mat[1][2] * mat[0][0];
        mat[2][2] = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];

        invsqrt = 1.0f / sqrtf(mat[2][0] * mat[2][0] + mat[2][1] * mat[2][1] + mat[2][2] * mat[2][2]);

        mat[2][0] *= invsqrt;
        mat[2][1] *= invsqrt;
        mat[2][2] *= invsqrt;

        mat[3][0] = platformPos[0];
        mat[3][1] = platformPos[1];
        mat[3][2] = platformPos[2];
        mat[0][3] = 0.0f;
        mat[1][3] = 0.0f;
        mat[2][3] = 0.0f;
        mat[3][3] = 1.0f;

        for (int h = 0; h < 2; h++) {
            for (int i = 0; i < 3; i++) {
                float vx = defaultTriangles[h][i][0];
                float vy = defaultTriangles[h][i][1];
                float vz = defaultTriangles[h][i][2];

                currentTriangles[h][i][0] = (short)(int)(vx * mat[0][0] + vy * mat[1][0] + vz * mat[2][0] + mat[3][0]);
                currentTriangles[h][i][1] = (short)(int)(vx * mat[0][1] + vy * mat[1][1] + vz * mat[2][1] + mat[3][1]);
                currentTriangles[h][i][2] = (short)(int)(vx * mat[0][2] + vy * mat[1][2] + vz * mat[2][2] + mat[3][2]);
            }

            triangleNormals[h][0] = ((currentTriangles[h][1][1] - currentTriangles[h][0][1]) * (currentTriangles[h][2][2] - currentTriangles[h][1][2])) - ((currentTriangles[h][1][2] - currentTriangles[h][0][2]) * (currentTriangles[h][2][1] - currentTriangles[h][1][1]));
            triangleNormals[h][1] = ((currentTriangles[h][1][2] - currentTriangles[h][0][2]) * (currentTriangles[h][2][0] - currentTriangles[h][1][0])) - ((currentTriangles[h][1][0] - currentTriangles[h][0][0]) * (currentTriangles[h][2][2] - currentTriangles[h][1][2]));
            triangleNormals[h][2] = ((currentTriangles[h][1][0] - currentTriangles[h][0][0]) * (currentTriangles[h][2][1] - currentTriangles[h][1][1])) - ((currentTriangles[h][1][1] - currentTriangles[h][0][1]) * (currentTriangles[h][2][0] - currentTriangles[h][1][0]));

            invsqrt = 1.0f / sqrtf(triangleNormals[h][0] * triangleNormals[h][0] + triangleNormals[h][1] * triangleNormals[h][1] + triangleNormals[h][2] * triangleNormals[h][2]);

            triangleNormals[h][0] *= invsqrt;
            triangleNormals[h][1] *= invsqrt;
            triangleNormals[h][2] *= invsqrt;
        }

        float returnPos[3] = { marioPos[0], marioPos[1], marioPos[2] };

        bool oTiltingPyramidMarioOnPlatform = false;
        bool onPlatform = false;

        float lastYNormal = triangleNormals[floor_idx][1];
        float lastPos[3] = { marioPos[0], marioPos[1], marioPos[2] };

        float landingPositions[3][3];
        float landingNormalsY[3];

        for (int f = 0; f < maxFrames; f++) {
            float dx;
            float dy;
            float dz;
            float d;

            float dist[3];
            float posBeforeRotation[3];
            float posAfterRotation[3];

            // Mario's position
            float mx;
            float my;
            float mz;

            int marioOnPlatform = 0;

            if (onPlatform)
            {
                mx = marioPos[0];
                my = marioPos[1];
                mz = marioPos[2];

                dist[0] = mx - (float)platformPos[0];
                dist[1] = my - (float)platformPos[1];
                dist[2] = mz - (float)platformPos[2];

                for (int i = 0; i < 3; i++) {
                    posBeforeRotation[i] = mat[0][i] * dist[0] + mat[1][i] * dist[1] + mat[2][i] * dist[2];
                }

                dx = mx - (float)platformPos[0];
                dy = 500.0f;
                dz = mz - (float)platformPos[2];
                d = sqrtf(dx * dx + dy * dy + dz * dz);

                //! Always true since dy = 500, making d >= 500.
                if (d != 0.0f) {
                    // Normalizing
                    d = 1.0 / d;
                    dx *= d;
                    dy *= d;
                    dz *= d;
                }
                else {
                    dx = 0.0f;
                    dy = 1.0f;
                    dz = 0.0f;
                }

                if (oTiltingPyramidMarioOnPlatform == true)
                    marioOnPlatform++;
                oTiltingPyramidMarioOnPlatform = true;
            }
            else
            {
                dx = 0.0f;
                dy = 1.0f;
                dz = 0.0f;
                oTiltingPyramidMarioOnPlatform = false;
            }

            // Approach the normals by 0.01f towards the new goal, then create a transform matrix and orient the object. 
            // Outside of the other conditionals since it needs to tilt regardless of whether Mario is on.

            normal[0] = (normal[0] <= dx) ? ((dx - normal[0] < 0.01f) ? dx : (normal[0] + 0.01f)) : ((dx - normal[0] > -0.01f) ? dx : (normal[0] - 0.01f));
            normal[1] = (normal[1] <= dy) ? ((dy - normal[1] < 0.01f) ? dy : (normal[1] + 0.01f)) : ((dy - normal[1] > -0.01f) ? dy : (normal[1] - 0.01f));
            normal[2] = (normal[2] <= dz) ? ((dz - normal[2] < 0.01f) ? dz : (normal[2] + 0.01f)) : ((dz - normal[2] > -0.01f) ? dz : (normal[2] - 0.01f));

            mat[1][0] = normal[0];
            mat[1][1] = normal[1];
            mat[1][2] = normal[2];

            invsqrt = 1.0f / sqrtf(mat[1][0] * mat[1][0] + mat[1][1] * mat[1][1] + mat[1][2] * mat[1][2]);

            mat[1][0] *= invsqrt;
            mat[1][1] *= invsqrt;
            mat[1][2] *= invsqrt;

            mat[0][0] = mat[1][1] * 1.0f - 0.0f * mat[1][2];
            mat[0][1] = mat[1][2] * 0.0f - 1.0f * mat[1][0];
            mat[0][2] = mat[1][0] * 0.0f - 0.0f * mat[1][1];

            invsqrt = 1.0f / sqrtf(mat[0][0] * mat[0][0] + mat[0][1] * mat[0][1] + mat[0][2] * mat[0][2]);

            mat[0][0] *= invsqrt;
            mat[0][1] *= invsqrt;
            mat[0][2] *= invsqrt;

            mat[2][0] = mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2];
            mat[2][1] = mat[0][2] * mat[1][0] - mat[1][2] * mat[0][0];
            mat[2][2] = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];

            invsqrt = 1.0f / sqrtf(mat[2][0] * mat[2][0] + mat[2][1] * mat[2][1] + mat[2][2] * mat[2][2]);

            mat[2][0] *= invsqrt;
            mat[2][1] *= invsqrt;
            mat[2][2] *= invsqrt;

            mat[3][0] = platformPos[0];
            mat[3][1] = platformPos[1];
            mat[3][2] = platformPos[2];
            mat[0][3] = 0.0f;
            mat[1][3] = 0.0f;
            mat[2][3] = 0.0f;
            mat[3][3] = 1.0f;

            for (int h = 0; h < 2; h++) {
                for (int i = 0; i < 3; i++) {
                    float vx = defaultTriangles[h][i][0];
                    float vy = defaultTriangles[h][i][1];
                    float vz = defaultTriangles[h][i][2];

                    currentTriangles[h][i][0] = (short)(int)(vx * mat[0][0] + vy * mat[1][0] + vz * mat[2][0] + mat[3][0]);
                    currentTriangles[h][i][1] = (short)(int)(vx * mat[0][1] + vy * mat[1][1] + vz * mat[2][1] + mat[3][1]);
                    currentTriangles[h][i][2] = (short)(int)(vx * mat[0][2] + vy * mat[1][2] + vz * mat[2][2] + mat[3][2]);
                }

                triangleNormals[h][0] = ((currentTriangles[h][1][1] - currentTriangles[h][0][1]) * (currentTriangles[h][2][2] - currentTriangles[h][1][2])) - ((currentTriangles[h][1][2] - currentTriangles[h][0][2]) * (currentTriangles[h][2][1] - currentTriangles[h][1][1]));
                triangleNormals[h][1] = ((currentTriangles[h][1][2] - currentTriangles[h][0][2]) * (currentTriangles[h][2][0] - currentTriangles[h][1][0])) - ((currentTriangles[h][1][0] - currentTriangles[h][0][0]) * (currentTriangles[h][2][2] - currentTriangles[h][1][2]));
                triangleNormals[h][2] = ((currentTriangles[h][1][0] - currentTriangles[h][0][0]) * (currentTriangles[h][2][1] - currentTriangles[h][1][1])) - ((currentTriangles[h][1][1] - currentTriangles[h][0][1]) * (currentTriangles[h][2][0] - currentTriangles[h][1][0]));

                invsqrt = 1.0f / sqrtf(triangleNormals[h][0] * triangleNormals[h][0] + triangleNormals[h][1] * triangleNormals[h][1] + triangleNormals[h][2] * triangleNormals[h][2]);

                triangleNormals[h][0] *= invsqrt;
                triangleNormals[h][1] *= invsqrt;
                triangleNormals[h][2] *= invsqrt;
            }

            // If Mario is on the platform, adjust his position for the platform tilt.
            if (marioOnPlatform) {
                for (int i = 0; i < 3; i++) {
                    posAfterRotation[i] = mat[0][i] * dist[0] + mat[1][i] * dist[1] + mat[2][i] * dist[2];
                }

                mx += posAfterRotation[0] - posBeforeRotation[0];
                my += posAfterRotation[1] - posBeforeRotation[1];
                mz += posAfterRotation[2] - posBeforeRotation[2];
                marioPos[0] = mx;
                marioPos[1] = my;
                marioPos[2] = mz;
            }

            floor_height = 0.0;
            floor_idx = -1;

            short x = (short)(int)marioPos[0];
            short y = (short)(int)marioPos[1];
            short z = (short)(int)marioPos[2];

            for (int i = 0; i < 2; i++) {
                short x1 = currentTriangles[i][0][0];
                short z1 = currentTriangles[i][0][2];
                short x2 = currentTriangles[i][1][0];
                short z2 = currentTriangles[i][1][2];

                // Check that the point is within the triangle bounds.
                if ((z1 - z) * (x2 - x1) - (x1 - x) * (z2 - z1) < 0) {
                    continue;
                }

                // To slightly save on computation time, set this later.
                int16_t x3 = currentTriangles[i][2][0];
                int16_t z3 = currentTriangles[i][2][2];

                if ((z2 - z) * (x3 - x2) - (x2 - x) * (z3 - z2) < 0) {
                    continue;
                }
                if ((z3 - z) * (x1 - x3) - (x3 - x) * (z1 - z3) < 0) {
                    continue;
                }

                float nx = triangleNormals[i][0];
                float ny = triangleNormals[i][1];
                float nz = triangleNormals[i][2];
                float oo = -(nx * x1 + ny * currentTriangles[i][0][1] + nz * z1);

                // Find the height of the floor at a given location.
                float height = -(x * nx + nz * z + oo) / ny;
                // Checks for floor interaction with a 78 unit buffer.
                if (y - (height + -78.0f) < 0.0f) {
                    continue;
                }

                floor_height = height;
                floor_idx = i;
                break;
            }

            if (f < 3) {
                if (floor_idx == -1) {
                    landingNormalsY[f] = 1.0;
                }
                else {
                    landingNormalsY[f] = triangleNormals[floor_idx][1];
                }

                landingPositions[f][0] = marioPos[0];
                landingPositions[f][1] = marioPos[1];
                landingPositions[f][2] = marioPos[2];
            }

            bool oldOnPlatform = onPlatform;
            onPlatform = floor_idx != -1 && fabsf(marioPos[1] - floor_height) <= 4.0;

            //Check if Mario is under the lava, or too far below the platform for it to conceivably be in reach later
            if ((floor_idx != -1 && floor_height <= -3071.0f) || (floor_idx != -1 && marioPos[1] - floor_height < -20.0f))
            {
                break;
            }

            if (onPlatform && oldOnPlatform) {
                float testNormal[3] = { fabs(normal[0]), fabs(normal[1]), fabs(normal[2]) };

                bool validSolution = false;

                if (testNormal[0] > testNormal[1] || testNormal[2] > testNormal[1]) {
                    validSolution = true;
                }
                else {
                    float offset = 0.01;

                    float a = testNormal[0] - offset;
                    float b = testNormal[2] - offset;
                    float c = testNormal[2];
                    float d = sqrtf(1 - testNormal[2] * testNormal[2]);
                    float sign = 1;

                    float v = testNormal[1] - offset;

                    float sqrt1 = sqrtf(a * a + v * v);
                    float sqrt2 = sqrtf(a * a + b * b + v * v);
                    float sqrt3 = sqrtf(testNormal[1] * testNormal[1] + testNormal[0] * testNormal[0]);
                    float sqrt4 = sqrtf(testNormal[1] * testNormal[1] + testNormal[0] * testNormal[0] + testNormal[2] * testNormal[2]);

                    float result = sign * d * sqrt1 * sqrt3 * (d * sqrt2 * (sqrt1 * testNormal[0] - a * sqrt3) * sqrt4 + c * (-sqrt1 * sqrt2 * testNormal[1] * testNormal[2] + b * v * sqrt3 * sqrt4));

                    if (result < 0) {
                        validSolution = true;
                    }
                    else {
                        c = sqrtf(1 - testNormal[0] * testNormal[0]);
                        d = testNormal[0];
                        sign = -1;
                        result = sign * d * sqrt1 * sqrt3 * (d * sqrt2 * (sqrt1 * testNormal[0] - a * sqrt3) * sqrt4 + c * (-sqrt1 * sqrt2 * testNormal[1] * testNormal[2] + b * v * sqrt3 * sqrt4));

                        if (result < 0) {
                            validSolution = true;
                        }
                    }
                }

                if (validSolution) {
                    int solIdx = atomicAdd(&nPlatSolutions, 1);

                    if (solIdx < MAX_PLAT_SOLUTIONS) {
                        struct PlatformSolution solution;
                        solution.endNormal[0] = normal[0];
                        solution.endNormal[1] = normal[1];
                        solution.endNormal[2] = normal[2];
                        solution.endPosition[0] = marioPos[0];
                        solution.endPosition[1] = marioPos[1];
                        solution.endPosition[2] = marioPos[2];
                        solution.endFloorIdx = floor_idx;
                        solution.returnPosition[0] = returnPos[0];
                        solution.returnPosition[1] = returnPos[1];
                        solution.returnPosition[2] = returnPos[2];
                        solution.nFrames = f;
                        solution.penultimateFloorNormalY = lastYNormal;
                        solution.penultimatePosition[0] = lastPos[0];
                        solution.penultimatePosition[1] = lastPos[1];
                        solution.penultimatePosition[2] = lastPos[2];
                        for (int j = 0; j < 2; j++) {
                            solution.endTriangleNormals[j][0] = triangleNormals[j][0];
                            solution.endTriangleNormals[j][1] = triangleNormals[j][1];
                            solution.endTriangleNormals[j][2] = triangleNormals[j][2];

                            for (int k = 0; k < 3; k++) {
                                solution.endTriangles[j][k][0] = currentTriangles[j][k][0];
                                solution.endTriangles[j][k][1] = currentTriangles[j][k][1];
                                solution.endTriangles[j][k][2] = currentTriangles[j][k][2];
                            }
                        }
                        for (int f = 0; f < 3; f++) {
                            solution.landingPositions[f][0] = landingPositions[f][0];
                            solution.landingPositions[f][1] = landingPositions[f][1];
                            solution.landingPositions[f][2] = landingPositions[f][2];
                            solution.landingFloorNormalsY[f] = landingNormalsY[f];
                        }

                        platSolutions[solIdx] = solution;
                    }
                }
            }

            lastYNormal = triangleNormals[floor_idx][1];
            lastPos[0] = marioPos[0];
            lastPos[1] = marioPos[1];
            lastPos[2] = marioPos[2];
        }
    }
}

__global__ void testEdge(const float x0, const float x1, const float z0, const float z1, float normalX, float normalY, float normalZ, int maxFrames) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = blockDim.x * gridDim.x;

    double t = (double)idx / (double)total;

    float marioPos[3] = { x0 + t * (x1 - x0), -2500.0f, z0 + t * (z1 - z0) };
    float normal[3] = { normalX, normalY, normalZ };

    try_position(marioPos, normal, maxFrames);
}

__global__ void cudaFunc(const float minX, const float deltaX, const float minZ, const float deltaZ, const int width, const int height, float normalX, float normalY, float normalZ, int maxFrames) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < width * height) {
        float marioPos[3] = { minX - fmodf(minX, deltaX) + deltaX * (idx % width), -2500.0f, minZ - fmodf(minZ, deltaZ) + deltaZ * (idx / width) };
        float normal[3] = { normalX, normalY, normalZ };

        try_position(marioPos, normal, maxFrames);
    }
}

__global__ void try_stick_positionG() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSK5Solutions, MAX_SK_PHASE_FIVE)) {
        struct SKPhase5* sol5 = &(sk5Solutions[idx]);
        struct SKPhase4* sol4 = &(sk4Solutions[sol5->p4Idx]);
        struct SKPhase3* sol3 = &(sk3Solutions[sol4->p3Idx]);
        struct SKPhase2* sol2 = (sol3->p2Type / 2 == 0) ? ((sol3->p2Type % 2 == 0) ? &(sk2ASolutions[sol3->p2Idx]) : &(sk2BSolutions[sol3->p2Idx])) : ((sol3->p2Type % 2 == 0) ? &(sk2CSolutions[sol3->p2Idx]) : &(sk2DSolutions[sol3->p2Idx]));
        struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);

        float mag = sqrtf((float)(sol5->stickX * sol5->stickX + sol5->stickY * sol5->stickY));

        float xS = sol5->stickX;
        float yS = sol5->stickY;

        if (mag > 64.0f) {
            xS = xS * (64.0f / mag);
            yS = yS * (64.0f / mag);
            mag = 64.0f;
        }

        float intendedMag = ((mag / 64.0f) * (mag / 64.0f)) * 32.0f;
        int intendedYaw = atan2sG(-yS, xS) + sol4->cameraYaw;
        int intendedDYaw = intendedYaw - sol5->f1Angle;
        intendedDYaw = (65536 + (intendedDYaw % 65536)) % 65536;

        float lower10KSpeed = sol4->minPre10KSpeed;
        float upper10KSpeed = sol4->maxPre10KSpeed;

        float forward = gCosineTableG[intendedDYaw >> 4];
        forward *= 0.5f + 0.5f * lower10KSpeed / 100.0f;
        float lossFactor = intendedMag / 32.0f * forward * 0.02f + 0.92f;

        lower10KSpeed *= lossFactor;
        forward = gCosineTableG[intendedDYaw >> 4];
        forward *= 0.5f + 0.5f * upper10KSpeed / 100.0f;
        lossFactor = intendedMag / 32.0f * forward * 0.02f + 0.92f;

        upper10KSpeed *= lossFactor;

        lower10KSpeed = fminf(sol4->minPost10KSpeed, lower10KSpeed);
        upper10KSpeed = fmaxf(sol4->maxPost10KSpeed, upper10KSpeed);

        if (lower10KSpeed >= upper10KSpeed) {
            float xVel = gSineTableG[sol2->f2Angle >> 4];
            float zVel = gCosineTableG[sol2->f2Angle >> 4];

            xVel += zVel * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;
            zVel -= xVel * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;

            double f3Angle = 65536.0 * atan2(xVel, zVel) / (2.0 * M_PI);

            double angleDiff = fmod(65536.0 + f3Angle - sol2->f2Angle, 65536.0);
            angleDiff = fmod(angleDiff + 32768.0, 65536.0) - 32768.0;

            if (angleDiff >= sol4->minAngleDiff && angleDiff <= sol4->maxAngleDiff) {
                double w = intendedMag * gCosineTableG[intendedDYaw >> 4];
                double eqB = (50.0 + 147200.0 / w);
                double eqC = -(320000.0 / w) * lower10KSpeed;
                double eqDet = eqB * eqB - eqC;
                float minSpeed = sqrt(eqDet) - eqB;

                eqC = -(320000.0 / w) * upper10KSpeed;
                eqDet = eqB * eqB - eqC;
                float maxSpeed = sqrt(eqDet) - eqB;

                int solIdx = atomicAdd(&nSK6Solutions, 1);

                if (solIdx < MAX_SK_PHASE_SIX) {
                    struct SKPhase6* solution = &(sk6Solutions[solIdx]);
                    solution->p5Idx = idx;
                    solution->minPre10KSpeed = minSpeed;
                    solution->maxPre10KSpeed = maxSpeed;
                    solution->minPost10KSpeed = lower10KSpeed;
                    solution->maxPost10KSpeed = upper10KSpeed;
                    solution->angleDiff = angleDiff;
                }
            }
        }
    }
}

__global__ void try_slide_kick_routeG2() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSK4Solutions, MAX_SK_PHASE_FOUR)) {
        struct SKPhase4* sol4 = &(sk4Solutions[idx]);
        struct SKPhase3* sol3 = &(sk3Solutions[sol4->p3Idx]);
        struct SKPhase2* sol2 = (sol3->p2Type / 2 == 0) ? ((sol3->p2Type % 2 == 0) ? &(sk2ASolutions[sol3->p2Idx]) : &(sk2BSolutions[sol3->p2Idx])) : ((sol3->p2Type % 2 == 0) ? &(sk2CSolutions[sol3->p2Idx]) : &(sk2DSolutions[sol3->p2Idx]));
        struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);

        double minStickX = INFINITY;
        double maxStickX = -INFINITY;
        double minStickY = INFINITY;
        double maxStickY = -INFINITY;

        for (int j = sol1->minF1AngleIdx; j <= sol1->maxF1AngleIdx; j++) {
            int f1Angle = (65536 + gArctanTableG[j % 8192]) % 65536;

            for (int i = 0; i < 4; i++) {
                double m1;
                double n1;

                if (i % 2 == 0) {
                    m1 = sol4->minM1;
                }
                else {
                    m1 = sol4->maxM1;
                }

                if (i / 2 == 0) {
                    n1 = sol4->minN1;
                }
                else {
                    n1 = sol4->maxN1;
                }

                double targetDYaw = 65536.0 * (atan2(n1, m1) / (2.0 * M_PI));
                double targetMag = sqrtf(m1 * m1 + n1 * n1);

                double stickAngle = fmod(65536.0 + fmod(targetDYaw + f1Angle - sol4->cameraYaw, 65536.0), 65536.0);
                double stickMagnitude = sqrt(128.0 * targetMag);

                double xS;
                double yS;

                if (stickMagnitude < 64.0) {
                    yS = -stickMagnitude * cos(2.0 * M_PI * (stickAngle / 65536.0));
                    xS = stickMagnitude * sin(2.0 * M_PI * (stickAngle / 65536.0));

                    minStickX = fmin(minStickX, xS);
                    minStickY = fmin(minStickY, yS);
                    maxStickX = fmax(maxStickX, xS);
                    maxStickY = fmax(maxStickY, yS);
                }
                else {
                    if (stickAngle <= 8192.0 || stickAngle > 57344.0) {
                        yS = -122.0;
                        xS = -122.0 * tan(2.0 * M_PI * (stickAngle / 65536.0));

                        minStickX = fmin(minStickX, xS);
                        minStickY = fmin(minStickY, yS);

                        yS = 64.0 * cos(2.0 * M_PI * (stickAngle / 65536.0));
                        xS = 64.0 * sin(2.0 * M_PI * (stickAngle / 65536.0));

                        maxStickX = fmax(maxStickX, xS);
                        maxStickY = fmax(maxStickY, yS);
                    }
                    else if (stickAngle > 8192.0 && stickAngle <= 24576.0) {
                        yS = 64.0 * cos(2.0 * M_PI * (stickAngle / 65536.0));
                        xS = 64.0 * sin(2.0 * M_PI * (stickAngle / 65536.0));

                        minStickX = fmin(minStickX, xS);
                        minStickY = fmin(minStickY, yS);

                        yS = 121.0 / tan(2.0 * M_PI * (stickAngle / 65536.0));
                        xS = 121.0;

                        maxStickX = fmax(maxStickX, xS);
                        maxStickY = fmax(maxStickY, yS);
                    }
                    else if (stickAngle > 24576.0 && stickAngle <= 40960.0) {
                        yS = 64.0 * cos(2.0 * M_PI * (stickAngle / 65536.0));
                        xS = 64.0 * sin(2.0 * M_PI * (stickAngle / 65536.0));

                        minStickX = fmin(minStickX, xS);
                        minStickY = fmin(minStickY, yS);

                        yS = 121.0;
                        xS = 121.0 * tan(2.0 * M_PI * (stickAngle / 65536.0));

                        maxStickX = fmax(maxStickX, xS);
                        maxStickY = fmax(maxStickY, yS);
                    }
                    else {
                        yS = -122.0 / tan(2.0 * M_PI * (stickAngle / 65536.0));
                        xS = -122.0;

                        minStickX = fmin(minStickX, xS);
                        minStickY = fmin(minStickY, yS);

                        yS = 64.0 * cos(2.0 * M_PI * (stickAngle / 65536.0));
                        xS = 64.0 * sin(2.0 * M_PI * (stickAngle / 65536.0));

                        maxStickX = fmax(maxStickX, xS);
                        maxStickY = fmax(maxStickY, yS);
                    }
                }

                if (maxStickX - minStickX < maxStickY - minStickY) {
                    for (int x = (int)ceil(minStickX); x <= (int)floor(maxStickX); x++) {
                        if (x != 1) {
                            int y = (int)round(((double)x - minStickX) * (maxStickY - minStickY) / (maxStickX - minStickX) + minStickY);

                            if (y != 1) {
                                int solIdx = atomicAdd(&nSK5Solutions, 1);

                                if (solIdx < MAX_SK_PHASE_FIVE) {
                                    struct SKPhase5* solution = &(sk5Solutions[solIdx]);
                                    solution->p4Idx = idx;
                                    solution->stickX = x;
                                    solution->stickY = y;
                                    solution->f1Angle = f1Angle;
                                }
                            }
                        }
                    }
                }
                else {
                    for (int y = (int)ceil(minStickY); y <= (int)floor(maxStickY); y++) {
                        if (y != 1) {
                            int x = (int)round(((double)y - minStickY) * (maxStickX - minStickX) / (maxStickY - minStickY) + minStickX);

                            if (x != 1) {
                                int solIdx = atomicAdd(&nSK5Solutions, 1);

                                if (solIdx < MAX_SK_PHASE_FIVE) {
                                    struct SKPhase5* solution = &(sk5Solutions[solIdx]);
                                    solution->p4Idx = idx;
                                    solution->stickX = x;
                                    solution->stickY = y;
                                    solution->f1Angle = f1Angle;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void try_slide_kick_routeG(short* pyramidFloorPoints, const int nPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSK3Solutions, MAX_SK_PHASE_THREE)) {
        struct SKPhase3* sol3 = &(sk3Solutions[idx]);
        struct SKPhase2* sol2 = (sol3->p2Type / 2 == 0) ? ((sol3->p2Type % 2 == 0) ? &(sk2ASolutions[sol3->p2Idx]) : &(sk2BSolutions[sol3->p2Idx])) : ((sol3->p2Type % 2 == 0) ? &(sk2CSolutions[sol3->p2Idx]) : &(sk2DSolutions[sol3->p2Idx]));
        struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);

        double minF2Dist = INFINITY;
        double maxF2Dist = -INFINITY;

        for (int i = 0; i < nPoints; i++) {
            for (int j = 0; j < 4; j++) {
                double PX = 65536.0 * sol1->x1 + pyramidFloorPoints[3 * i];
                double PZ = 65536.0 * sol1->z1 + pyramidFloorPoints[3 * i + 2];
                double QX1 = 65536.0 * sol3->x2 + ((j / 2 == 0) ? tenKFloors[sol2->tenKFloorIdx][0] : tenKFloors[sol2->tenKFloorIdx][1]);
                double QX2 = 65536.0 * sol3->x2 + ((((j + 1) % 4) / 2 == 0) ? tenKFloors[sol2->tenKFloorIdx][0] : tenKFloors[sol2->tenKFloorIdx][1]);
                double QZ1 = 65536.0 * sol3->z2 + ((((j + 1) % 4) / 2 == 0) ? tenKFloors[sol2->tenKFloorIdx][3] : tenKFloors[sol2->tenKFloorIdx][2]);
                double QZ2 = 65536.0 * sol3->z2 + ((j / 2 == 0) ? tenKFloors[sol2->tenKFloorIdx][3] : tenKFloors[sol2->tenKFloorIdx][2]);

                double s = ((QZ1 - PZ) * gSineTableG[sol2->f2Angle >> 4] - (QX1 - PX) * gCosineTableG[sol2->f2Angle >> 4]) / ((QX2 - QX1) * gCosineTableG[sol2->f2Angle >> 4] - (QZ2 - QZ1) * gSineTableG[sol2->f2Angle >> 4]);

                if (s >= 0.0 && s <= 1.0) {
                    double dist;
                    
                    if (fabs(gSineTableG[sol2->f2Angle >> 4]) > fabs(gCosineTableG[sol2->f2Angle >> 4])) {
                        dist = (s * (QX2 - QX1) - (PX - QX1)) / gSineTableG[sol2->f2Angle >> 4];
                    }
                    else {
                        dist = (s * (QZ2 - QZ1) - (PZ - QZ1)) / gCosineTableG[sol2->f2Angle >> 4];
                    }

                    minF2Dist = fmin(minF2Dist, dist);
                    maxF2Dist = fmax(maxF2Dist, dist);
                }
            }
        }

        for (int i = 0; i < nPoints; i++) {
            for (int j = 0; j < 4; j++) {
                double PX = 65536.0 * sol3->x2 + ((j / 2 == 0) ? tenKFloors[sol2->tenKFloorIdx][0] : tenKFloors[sol2->tenKFloorIdx][1]);
                double PZ = 65536.0 * sol3->z2 + ((((j + 1) % 4) / 2 == 0) ? tenKFloors[sol2->tenKFloorIdx][0] : tenKFloors[sol2->tenKFloorIdx][1]);
                double QX1 = 65536.0 * sol1->x1 + pyramidFloorPoints[3 * i];
                double QX2 = 65536.0 * sol1->x1 + pyramidFloorPoints[3 * ((i + 1)%nPoints)];
                double QZ1 = 65536.0 * sol1->z1 + pyramidFloorPoints[3 * i + 2];
                double QZ2 = 65536.0 * sol1->z1 + pyramidFloorPoints[3 * ((i + 1) % nPoints) + 2];

                double s = ((QZ1 - PZ) * gSineTableG[sol2->f2Angle >> 4] - (QX1 - PX) * gCosineTableG[sol2->f2Angle >> 4]) / ((QX2 - QX1) * gCosineTableG[sol2->f2Angle >> 4] - (QZ2 - QZ1) * gSineTableG[sol2->f2Angle >> 4]);

                if (s >= 0.0 && s <= 1.0) {
                    double dist;

                    if (fabs(gSineTableG[sol2->f2Angle >> 4]) > fabs(gCosineTableG[sol2->f2Angle >> 4])) {
                        dist = -(s * (QX2 - QX1) - (PX - QX1)) / gSineTableG[sol2->f2Angle >> 4];
                    }
                    else {
                        dist = -(s * (QZ2 - QZ1) - (PZ - QZ1)) / gCosineTableG[sol2->f2Angle >> 4];
                    }

                    minF2Dist = fmin(minF2Dist, dist);
                    maxF2Dist = fmax(maxF2Dist, dist);
                }
            }
        }

        double minSpeed = fmaxf(sol1->minSpeed - 2.85f, 4.0 * minF2Dist / (float)sol1->q2);
        double maxSpeed = fminf(sol1->maxSpeed + 0.15f, 4.0 * maxF2Dist / (float)sol1->q2);

        if (minSpeed <= maxSpeed) {
            double minF3Dist = INFINITY;
            double maxF3Dist = -INFINITY;

            double minAngleDiff = INFINITY;
            double maxAngleDiff = -INFINITY;

            for (int i = 0; i < nPoints; i++) {
                for (int j = 0; j < 4; j++) {
                    double xDist;
                    double zDist;

                    if (j % 2 == 0) {
                        xDist = (65536.0 * sol3->x2 + tenKFloors[sol2->tenKFloorIdx][0]) - pyramidFloorPoints[3 * i];
                    }
                    else {
                        xDist = (65536.0 * sol3->x2 + tenKFloors[sol2->tenKFloorIdx][1]) - pyramidFloorPoints[3 * i];
                    }

                    if (j / 2 == 0) {
                        zDist = (65536.0 * sol3->z2 + tenKFloors[sol2->tenKFloorIdx][2]) - pyramidFloorPoints[3 * i + 2];
                    }
                    else {
                        zDist = (65536.0 * sol3->z2 + tenKFloors[sol2->tenKFloorIdx][3]) - pyramidFloorPoints[3 * i + 2];
                    }

                    double dist = sqrt(xDist * xDist + zDist * zDist);

                    minF3Dist = fmin(minF3Dist, dist);
                    maxF3Dist = fmax(maxF3Dist, dist);

                    double f3Angle = 65536.0 * atan2(xDist, zDist) / (2.0 * M_PI);

                    double angleDiff = fmod(65536.0 + f3Angle - sol2->f2Angle, 65536.0);
                    angleDiff = fmod(angleDiff + 32768.0, 65536.0) - 32768.0;

                    minAngleDiff = fmin(minAngleDiff, angleDiff);
                    maxAngleDiff = fmax(maxAngleDiff, angleDiff);
                }
            }

            minAngleDiff = fmax(minAngleDiff, -(double)maxF3Turn);
            maxAngleDiff = fmin(maxAngleDiff, (double)maxF3Turn);

            if (minAngleDiff <= maxAngleDiff) {
                double minF3Angle = minAngleDiff + sol2->f2Angle;
                double maxF3Angle = maxAngleDiff + sol2->f2Angle;

                double minN;
                double maxN;

                if (sol2->f2Angle == 0 || sol2->f2Angle == 32768) {
                    double sinF2Angle = sin(2.0 * M_PI * (double)sol2->f2Angle / 65536.0);

                    minN = -cos(2.0 * M_PI * minF3Angle / 65536.0) / sinF2Angle;
                    maxN = -cos(2.0 * M_PI * maxF3Angle / 65536.0) / sinF2Angle;
                }
                else {
                    double sinF2Angle = gSineTableG[sol2->f2Angle >> 4];
                    double cosF2Angle = gCosineTableG[sol2->f2Angle >> 4];

                    double sinMinF3Angle = sin(2.0 * M_PI * minF3Angle / 65536.0);
                    double cosMinF3Angle = cos(2.0 * M_PI * minF3Angle / 65536.0);

                    double sinMaxF3Angle = sin(2.0 * M_PI * maxF3Angle / 65536.0);
                    double cosMaxF3Angle = cos(2.0 * M_PI * maxF3Angle / 65536.0);

                    double t = sinF2Angle / cosF2Angle;
                    double s = sinMinF3Angle / cosMinF3Angle;

                    bool signTest = (cosF2Angle > 0 && cosMinF3Angle > 0) || (cosF2Angle < 0 && cosMinF3Angle < 0);

                    if (signTest) {
                        minN = (-((double)s * (double)t) - 1.0 + sqrt(((double)s * (double)t - 1.0) * ((double)s * (double)t - 1.0) + 4.0 * (double)s * (double)s)) / (2.0 * (double)s);
                    }
                    else {
                        minN = (-((double)s * (double)t) - 1.0 - sqrt(((double)s * (double)t - 1.0) * ((double)s * (double)t - 1.0) + 4.0 * (double)s * (double)s)) / (2.0 * (double)s);
                    }

                    s = sinMaxF3Angle / cosMaxF3Angle;

                    signTest = (cosF2Angle > 0 && cosMaxF3Angle > 0) || (cosF2Angle < 0 && cosMaxF3Angle < 0);

                    if (signTest) {
                        maxN = (-((double)s * (double)t) - 1.0 + sqrt(((double)s * (double)t - 1.0) * ((double)s * (double)t - 1.0) + 4.0 * (double)s * (double)s)) / (2.0 * (double)s);
                    }
                    else {
                        maxN = (-((double)s * (double)t) - 1.0 - sqrt(((double)s * (double)t - 1.0) * ((double)s * (double)t - 1.0) + 4.0 * (double)s * (double)s)) / (2.0 * (double)s);
                    }
                }

                double minN1 = 32.0 * minN / 0.05;
                double maxN1 = 32.0 * maxN / 0.05;

                if (minN1 > maxN1) {
                    double temp = minN1;
                    minN1 = maxN1;
                    maxN1 = temp;
                }

                minN1 = fmax(minN1, -32.0);
                maxN1 = fmin(maxN1, 32.0);

                if (minN1 <= maxN1) {
                    float minPost10KSpeed = -4.0 * minF3Dist / tenKFloors[sol2->tenKFloorIdx][7];
                    float maxPost10KSpeed = -4.0 * maxF3Dist / tenKFloors[sol2->tenKFloorIdx][7];

                    double minM = (double)minPost10KSpeed / (double)maxSpeed;
                    double maxM = (double)maxPost10KSpeed / (double)minSpeed;

                    double minM1 = 32.0 * ((minM - 0.92) / 0.02) / (double)(0.5f + (0.5f * maxSpeed / 100.0f));
                    double maxM1 = 32.0 * ((maxM - 0.92) / 0.02) / (double)(0.5f + (0.5f * minSpeed / 100.0f));

                    if (minM1 > maxM1) {
                        double temp = minM1;
                        minM1 = maxM1;
                        maxM1 = temp;
                    }

                    minM1 = fmax(minM1, -32.0);
                    maxM1 = fmin(maxM1, 0.0);

                    if (minM1 <= maxM1) {
                        float cameraFocus[3] = { 0.0f, 0.0f, 0.0f };

                        for (int i = 0; i < nPoints; i++) {
                            cameraFocus[0] += pyramidFloorPoints[3 * i];
                            cameraFocus[1] += pyramidFloorPoints[3 * i + 1];
                            cameraFocus[2] += pyramidFloorPoints[3 * i + 2];
                        }

                        cameraFocus[0] /= nPoints;
                        cameraFocus[1] /= nPoints;
                        cameraFocus[2] /= nPoints;

                        cameraFocus[0] += 0.8 * 65536.0 * sol1->x1;
                        cameraFocus[2] += 0.8 * 65536.0 * sol1->z1;

                        float distToCamera = sqrtf(cameraFocus[0] * cameraFocus[0] + cameraFocus[2] * cameraFocus[2] - 1073741824.0f);
                        float cameraPosition1[3];
                        cameraPosition1[0] = 32768.0f * (32768.0f * cameraFocus[0] + distToCamera * cameraFocus[2]) / (distToCamera * distToCamera + 1073741824.0f);
                        cameraPosition1[1] = -2918.0f;
                        cameraPosition1[2] = 32768.0f * (32768.0f * cameraFocus[2] - distToCamera * cameraFocus[0]) / (distToCamera * distToCamera + 1073741824.0f);

                        float cameraPosition2[3];
                        cameraPosition2[0] = 32768.0f * (32768.0f * cameraFocus[0] - distToCamera * cameraFocus[2]) / (distToCamera * distToCamera + 1073741824.0f);
                        cameraPosition2[1] = -2918.0f;
                        cameraPosition2[2] = 32768.0f * (distToCamera * cameraFocus[0] + 32768.0f * cameraFocus[2]) / (distToCamera * distToCamera + 1073741824.0f);

                        int minCameraYaw = calculate_camera_yaw(cameraFocus, cameraPosition1, sol2->f2Angle);
                        int maxCameraYaw = calculate_camera_yaw(cameraFocus, cameraPosition2, sol2->f2Angle);

                        if ((short)(maxCameraYaw - minCameraYaw) < 0) {
                            int temp = minCameraYaw;
                            minCameraYaw = maxCameraYaw;
                            maxCameraYaw = temp;
                        }

                        int minCameraIdx = gReverseArctanTableG[(65536 + minCameraYaw) % 65536];
                        int maxCameraIdx = gReverseArctanTableG[(65536 + maxCameraYaw) % 65536];

                        if (minCameraIdx > maxCameraIdx) {
                            maxCameraIdx += 8192;
                        }

                        for (int cIdx = minCameraIdx; cIdx <= maxCameraIdx; cIdx++) {
                            int cameraYaw = gArctanTableG[(8192 + cIdx) % 8192];
                            cameraYaw = (65536 + cameraYaw) % 65536;

                            if (validCameraAngle[cameraYaw]) {
                                int solIdx = atomicAdd(&nSK4Solutions, 1);

                                if (solIdx < MAX_SK_PHASE_FOUR) {
                                    struct SKPhase4* solution = &(sk4Solutions[solIdx]);
                                    solution->p3Idx = idx;
                                    solution->cameraYaw = cameraYaw;
                                    solution->minM1 = minM1;
                                    solution->maxM1 = maxM1;
                                    solution->minN1 = minN1;
                                    solution->maxN1 = maxN1;
                                    solution->minPre10KSpeed = minSpeed;
                                    solution->maxPre10KSpeed = maxSpeed;
                                    solution->minPost10KSpeed = minPost10KSpeed;
                                    solution->maxPost10KSpeed = maxPost10KSpeed;
                                    solution->minAngleDiff = minAngleDiff;
                                    solution->maxAngleDiff = maxAngleDiff;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void find_slide_kick_setupG3a(float platformMinZ, float platformMaxZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSK2ASolutions, MAX_SK_PHASE_TWO_A)) {
        struct SKPhase2* sol2 = &(sk2CSolutions[idx]);
        struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);

        int cosSign = (sol2->cosAngle > 0) - (sol2->cosAngle < 0);

        double speed1 = ((cosSign + 1) >> 1) * (sol1->minSpeed - 2.85f) + (((cosSign + 1) >> 1) ^ 1) * (sol1->maxSpeed + 0.15f);
        double speed2 = ((cosSign + 1) >> 1) * (sol1->maxSpeed + 0.15f) + (((cosSign + 1) >> 1) ^ 1) * (sol1->minSpeed - 2.85f);

        int minF2ZPU = (int)ceil((65536.0 * sol1->z1 + platformMinZ + speed1 * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][3]) / 65536.0);
        int maxF2ZPU = (int)floor((65536.0 * sol1->z1 + platformMaxZ + speed2 * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][2]) / 65536.0);

        minF2ZPU += (sol1->q2 + ((sol1->z1 - minF2ZPU) % sol1->q2)) % sol1->q2;
        maxF2ZPU -= (sol1->q2 + ((minF2ZPU - sol1->z1) % sol1->q2)) % sol1->q2;

        for (int z2 = minF2ZPU; z2 <= maxF2ZPU; z2 += sol1->q2) {
            int solIdx = atomicAdd(&nSK3Solutions, 1);

            if (solIdx < MAX_SK_PHASE_THREE) {
                struct SKPhase3* solution = &(sk3Solutions[solIdx]);
                solution->p2Idx = idx;
                solution->p2Type = 0;
                solution->x2 = sol1->x1;
                solution->z2 = z2;
            }
        }
    }
}


__global__ void find_slide_kick_setupG3b(float platformMinX, float platformMaxX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSK2BSolutions, MAX_SK_PHASE_TWO_B)) {
        struct SKPhase2* sol2 = &(sk2CSolutions[idx]);
        struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);

        int sinSign = (sol2->sinAngle > 0) - (sol2->sinAngle < 0);

        double speed1 = ((sinSign + 1) >> 1) * (sol1->minSpeed - 2.85f) + (((sinSign + 1) >> 1) ^ 1) * (sol1->maxSpeed + 0.15f);
        double speed2 = ((sinSign + 1) >> 1) * (sol1->maxSpeed + 0.15f) + (((sinSign + 1) >> 1) ^ 1) * (sol1->minSpeed - 2.85f);

        int minF2XPU = (int)ceil((65536.0 * sol1->x1 + platformMinX + speed1 * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][1]) / 65536.0);
        int maxF2XPU = (int)floor((65536.0 * sol1->x1 + platformMaxX + speed2 * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][0]) / 65536.0);

        minF2XPU += (sol1->q2 + ((sol1->x1 - minF2XPU) % sol1->q2)) % sol1->q2;
        maxF2XPU -= (sol1->q2 + ((minF2XPU - sol1->x1) % sol1->q2)) % sol1->q2;

        for (int x2 = minF2XPU; x2 <= maxF2XPU; x2 += sol1->q2) {
            int solIdx = atomicAdd(&nSK3Solutions, 1);

            if (solIdx < MAX_SK_PHASE_THREE) {
                struct SKPhase3* solution = &(sk3Solutions[solIdx]);
                solution->p2Idx = idx;
                solution->p2Type = 1;
                solution->x2 = x2;
                solution->z2 = sol1->z1;
            }
        }
    }
}

__global__ void find_slide_kick_setupG3c(float platformMinX, float platformMaxX, float platformMinZ, float platformMaxZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSK2CSolutions, MAX_SK_PHASE_TWO_C)) {
        struct SKPhase2* sol2 = &(sk2CSolutions[idx]);
        struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);

        double cotAngle = sol2->cosAngle / sol2->sinAngle;

        int sinSign = (sol2->sinAngle > 0) - (sol2->sinAngle < 0);
        int cosSign = (sol2->cosAngle > 0) - (sol2->cosAngle < 0);
        int cotSign = (cotAngle > 0) - (cotAngle < 0);

        double speed1 = ((sinSign + 1) >> 1) * (sol1->minSpeed - 2.85f) + (((sinSign + 1) >> 1) ^ 1) * (sol1->maxSpeed + 0.15f);
        double speed2 = ((sinSign + 1) >> 1) * (sol1->maxSpeed + 0.15f) + (((sinSign + 1) >> 1) ^ 1) * (sol1->minSpeed - 2.85f);

        int minF2XPU = (int)ceil((65536.0 * sol1->x1 + platformMinX + speed1 * sol2->sinAngle * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][1]) / 65536.0);
        int maxF2XPU = (int)floor((65536.0 * sol1->x1 + platformMaxX + speed2 * sol2->sinAngle * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][0]) / 65536.0);

        minF2XPU += (sol1->q2 + ((sol1->x1 - minF2XPU) % sol1->q2)) % sol1->q2;
        maxF2XPU -= (sol1->q2 + ((minF2XPU - sol1->x1) % sol1->q2)) % sol1->q2;

        speed1 = ((cosSign + 1) >> 1) * (sol1->minSpeed - 2.85f) + (((cosSign + 1) >> 1) ^ 1) * (sol1->maxSpeed + 0.15f);
        speed2 = ((cosSign + 1) >> 1) * (sol1->maxSpeed + 0.15f) + (((cosSign + 1) >> 1) ^ 1) * (sol1->minSpeed - 2.85f);

        int minF2ZPU = (int)ceil((65536.0 * sol1->z1 + platformMinZ + speed1 * sol2->cosAngle * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][3]) / 65536.0);
        int maxF2ZPU = (int)floor((65536.0 * sol1->z1 + platformMaxZ + speed2 * sol2->cosAngle * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][2]) / 65536.0);

        minF2ZPU += (sol1->q2 + ((sol1->z1 - minF2ZPU) % sol1->q2)) % sol1->q2;
        maxF2ZPU -= (sol1->q2 + ((minF2ZPU - sol1->z1) % sol1->q2)) % sol1->q2;

        int floorPointIdx = 1 - ((cotSign + 1) >> 1);
        float tenKFloorX1 = tenKFloors[sol2->tenKFloorIdx][floorPointIdx];
        float tenKFloorX2 = tenKFloors[sol2->tenKFloorIdx][1 - floorPointIdx];
        float platformX1 = ((cotSign + 1) >> 1) * platformMaxX + (((cotSign + 1) >> 1) ^ 1) * platformMinX;
        float platformX2 = ((cotSign + 1) >> 1) * platformMinX + (((cotSign + 1) >> 1) ^ 1) * platformMaxX;
        float zRange1 = ((cotSign + 1) >> 1) * sol2->lower + (((cotSign + 1) >> 1) ^ 1) * sol2->upper;
        float zRange2 = ((cotSign + 1) >> 1) * sol2->upper + (((cotSign + 1) >> 1) ^ 1) * sol2->lower;

        for (int x2 = minF2XPU; x2 <= maxF2XPU; x2 += sol1->q2) {
            int minF2XZPU = (int)ceil((65536.0 * sol1->z1 + platformMinZ + ((65536.0 * x2 + tenKFloorX1) - (65536.0 * sol1->x1 + platformX1)) * cotAngle + zRange1 - tenKFloors[sol2->tenKFloorIdx][3]) / 65536.0);
            int maxF2XZPU = (int)floor((65536.0 * sol1->z1 + platformMaxZ + ((65536.0 * x2 + tenKFloorX2) - (65536.0 * sol1->x1 + platformX2)) * cotAngle + zRange2 - tenKFloors[sol2->tenKFloorIdx][2]) / 65536.0);

            minF2XZPU += (sol1->q2 + ((sol1->z1 - minF2XZPU) % sol1->q2)) % sol1->q2;
            maxF2XZPU -= (sol1->q2 + ((maxF2XZPU - sol1->z1) % sol1->q2)) % sol1->q2;

            minF2XZPU = max(minF2XZPU, minF2ZPU);
            maxF2XZPU = min(maxF2XZPU, maxF2ZPU);

            for (int z2 = minF2ZPU; z2 <= maxF2ZPU; z2 += sol1->q2) {
                int solIdx = atomicAdd(&nSK3Solutions, 1);

                if (solIdx < MAX_SK_PHASE_THREE) {
                    struct SKPhase3* solution = &(sk3Solutions[solIdx]);
                    solution->p2Idx = idx;
                    solution->p2Type = 2;
                    solution->x2 = x2;
                    solution->z2 = z2;
                }
            }
        }
    }
}

__global__ void find_slide_kick_setupG3d(float platformMinX, float platformMaxX, float platformMinZ, float platformMaxZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSK2DSolutions, MAX_SK_PHASE_TWO_D)) {
        struct SKPhase2* sol2 = &(sk2DSolutions[idx]);
        struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);

        double tanAngle = sol2->sinAngle / sol2->cosAngle;

        int sinSign = (sol2->sinAngle > 0) - (sol2->sinAngle < 0);
        int cosSign = (sol2->cosAngle > 0) - (sol2->cosAngle < 0);
        int tanSign = (tanAngle > 0) - (tanAngle < 0);

        double speed1 = ((sinSign + 1) >> 1) * (sol1->minSpeed - 2.85f) + (((sinSign + 1) >> 1) ^ 1) * (sol1->maxSpeed + 0.15f);
        double speed2 = ((sinSign + 1) >> 1) * (sol1->maxSpeed + 0.15f) + (((sinSign + 1) >> 1) ^ 1) * (sol1->minSpeed - 2.85f);

        int minF2XPU = (int)ceil((65536.0 * sol1->x1 + platformMinX + speed1 * sol2->sinAngle * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][1]) / 65536.0);
        int maxF2XPU = (int)floor((65536.0 * sol1->x1 + platformMaxX + speed2 * sol2->sinAngle * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][0]) / 65536.0);

        minF2XPU += (sol1->q2 + ((sol1->x1 - minF2XPU) % sol1->q2)) % sol1->q2;
        maxF2XPU -= (sol1->q2 + ((minF2XPU - sol1->x1) % sol1->q2)) % sol1->q2;

        speed1 = ((cosSign + 1) >> 1) * (sol1->minSpeed - 2.85f) + (((cosSign + 1) >> 1) ^ 1) * (sol1->maxSpeed + 0.15f);
        speed2 = ((cosSign + 1) >> 1) * (sol1->maxSpeed + 0.15f) + (((cosSign + 1) >> 1) ^ 1) * (sol1->minSpeed - 2.85f);

        int minF2ZPU = (int)ceil((65536.0 * sol1->z1 + platformMinZ + speed1 * sol2->cosAngle * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][3]) / 65536.0);
        int maxF2ZPU = (int)floor((65536.0 * sol1->z1 + platformMaxZ + speed2 * sol2->cosAngle * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][2]) / 65536.0);

        minF2ZPU += (sol1->q2 + ((sol1->z1 - minF2ZPU) % sol1->q2)) % sol1->q2;
        maxF2ZPU -= (sol1->q2 + ((minF2ZPU - sol1->z1) % sol1->q2)) % sol1->q2;

        int floorPointIdx = 3 - ((tanSign + 1) >> 1);
        float tenKFloorZ1 = tenKFloors[sol2->tenKFloorIdx][floorPointIdx];
        float tenKFloorZ2 = tenKFloors[sol2->tenKFloorIdx][5 - floorPointIdx];
        float platformZ1 = ((tanSign + 1) >> 1) * platformMaxZ + (((tanSign + 1) >> 1) ^ 1) * platformMinZ;
        float platformZ2 = ((tanSign + 1) >> 1) * platformMinZ + (((tanSign + 1) >> 1) ^ 1) * platformMaxZ;
        float xRange1 = ((tanSign + 1) >> 1) * sol2->lower + (((tanSign + 1) >> 1) ^ 1) * sol2->upper;
        float xRange2 = ((tanSign + 1) >> 1) * sol2->upper + (((tanSign + 1) >> 1) ^ 1) * sol2->lower;

        for (int z2 = minF2ZPU; z2 <= maxF2ZPU; z2 += sol1->q2) {
            int minF2ZXPU = (int)ceil((65536.0 * sol1->x1 + platformMinX + ((65536.0 * z2 + tenKFloorZ1) - (65536.0 * sol1->z1 + platformZ1)) * tanAngle + xRange1 - tenKFloors[sol2->tenKFloorIdx][1]) / 65536.0);
            int maxF2ZXPU = (int)floor((65536.0 * sol1->x1 + platformMaxX + ((65536.0 * z2 + tenKFloorZ2) - (65536.0 * sol1->z1 + platformZ2)) * tanAngle + xRange2 - tenKFloors[sol2->tenKFloorIdx][0]) / 65536.0);

            minF2ZXPU += (sol1->q2 + ((sol1->x1 - minF2ZXPU) % sol1->q2)) % sol1->q2;
            maxF2ZXPU -= (sol1->q2 + ((maxF2ZXPU - sol1->x1) % sol1->q2)) % sol1->q2;

            minF2ZXPU = max(minF2ZXPU, minF2XPU);
            maxF2ZXPU = min(maxF2ZXPU, maxF2XPU);

            for (int x2 = minF2ZXPU; x2 <= maxF2ZXPU; x2 += sol1->q2) {
                int solIdx = atomicAdd(&nSK3Solutions, 1);

                if (solIdx < MAX_SK_PHASE_THREE) {
                    struct SKPhase3* solution = &(sk3Solutions[solIdx]);
                    solution->p2Idx = idx;
                    solution->p2Type = 3;
                    solution->x2 = x2;
                    solution->z2 = z2;
                }
            }
        }
    }
}

__global__ void find_slide_kick_setupG2(short* floorPoints, const int nPoints, float floorNormalY, float platformMinX, float platformMaxX, float platformMinZ, float platformMaxZ, float midPointX, float midPointZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSK1Solutions, MAX_SK_PHASE_ONE)) {
        struct SKPhase1 *sol = &(sk1Solutions[idx]);
        double puAngle = 65536.0 * atan2((double)sol->x1, (double)sol->z1) / (2.0 * M_PI);
        puAngle = fmod(65536.0 + puAngle, 65536.0);
        
        int puAngleClosest = (65536 + atan2sG(sol->z1, sol->x1)) % 65536;

        double sinMaxAngle = sin(2.0 * M_PI * (double)maxF3Turn / 65536.0);
        double maxF2AngleChange = fmod(32768.0 - (65536.0 * asin(sol->q2 * sinMaxAngle / (4.0 * floorNormalY)) / (2.0 * M_PI)) - maxF3Turn, 32768.0);
        maxF2AngleChange = fabs(fmod(maxF2AngleChange + 16384.0, 32768.0) - 16384.0);

        int minF2AngleIdx = gReverseArctanTableG[puAngleClosest];
        int maxF2AngleIdx = gReverseArctanTableG[puAngleClosest];

        while ((65536 + puAngleClosest - ((gArctanTableG[(minF2AngleIdx + 8191) % 8192] >> 4) << 4)) % 65536 < maxF2AngleChange) {
            minF2AngleIdx = minF2AngleIdx - 1;
        }

        while ((65536 + ((gArctanTableG[(maxF2AngleIdx + 1) % 8192] >> 4) << 4) - puAngleClosest) % 65536 < maxF2AngleChange) {
            maxF2AngleIdx = maxF2AngleIdx + 1;
        }

        for (int a = minF2AngleIdx; a <= maxF2AngleIdx; a++) {
            int f2Angle = gArctanTableG[(8192 + a) % 8192];
            f2Angle = (65536 + f2Angle) % 65536;

            if (f2Angle == 0 || f2Angle == 32768) {
                for (int i = 0; i < nTenKFloors; i++) {
                    float minX = fmaxf(platformMinX, tenKFloors[i][0]);
                    float maxX = fminf(platformMaxX, tenKFloors[i][1]);

                    if (minX < maxX) {
                        int solIdx = atomicAdd(&nSK2ASolutions, 1);

                        if (solIdx < MAX_SK_PHASE_TWO_A) {
                            struct SKPhase2* solution = &(sk2ASolutions[solIdx]);
                            solution->p1Idx = idx;
                            solution->f2Angle = 0;
                            solution->tenKFloorIdx = i;
                            solution->sinAngle = 0.0;
                            solution->cosAngle = 1.0;
                        }

                        solIdx = atomicAdd(&nSK2ASolutions, 1);

                        if (solIdx < MAX_SK_PHASE_TWO_A) {
                            struct SKPhase2* solution = &(sk2ASolutions[solIdx]);
                            solution->p1Idx = idx;
                            solution->f2Angle = 32768;
                            solution->tenKFloorIdx = i;
                            solution->sinAngle = 0.0;
                            solution->cosAngle = -1.0;
                        }
                    }
                }
            }
            else if (f2Angle == 16384 || f2Angle == 49152) {
                for (int i = 0; i < nTenKFloors; i++) {
                    float minZ = fmaxf(platformMinZ, tenKFloors[i][2]);
                    float maxZ = fminf(platformMaxZ, tenKFloors[i][3]);

                    if (minZ < maxZ) {
                        int solIdx = atomicAdd(&nSK2BSolutions, 1);

                        if (solIdx < MAX_SK_PHASE_TWO_B) {
                            struct SKPhase2* solution = &(sk2BSolutions[solIdx]);
                            solution->p1Idx = idx;
                            solution->f2Angle = 16384;
                            solution->tenKFloorIdx = i;
                            solution->sinAngle = 1.0;
                            solution->cosAngle = 0.0;
                        }

                        solIdx = atomicAdd(&nSK2BSolutions, 1);

                        if (solIdx < MAX_SK_PHASE_TWO_B) {
                            struct SKPhase2* solution = &(sk2BSolutions[solIdx]);
                            solution->p1Idx = idx;
                            solution->f2Angle = 49152;
                            solution->tenKFloorIdx = i;
                            solution->sinAngle = 1.0;
                            solution->cosAngle = 0.0;
                        }
                    }
                }
            }
            else {
                double sinAngle = gSineTableG[f2Angle >> 4];
                double cosAngle = gCosineTableG[f2Angle >> 4];

                if (fabs(sinAngle) < fabs(cosAngle)) {
                    float lowerZ = INFINITY;
                    float upperZ = -INFINITY;

                    double cotAngle = cosAngle / sinAngle;

                    for (int i = 0; i < nPoints; i++) {
                        float testZ = floorPoints[3 * i + 2] + ((midPointX - floorPoints[3 * i]) * cotAngle) - midPointZ;
                        lowerZ = fminf(lowerZ, testZ);
                        upperZ = fmaxf(upperZ, testZ);
                    }

                    for (int i = 0; i < nTenKFloors; i++) {
                        int solIdx = atomicAdd(&nSK2CSolutions, 1);

                        if (solIdx < MAX_SK_PHASE_TWO_C) {
                            struct SKPhase2* solution = &(sk2CSolutions[solIdx]);
                            solution->p1Idx = idx;
                            solution->f2Angle = f2Angle;
                            solution->tenKFloorIdx = i;
                            solution->lower = lowerZ;
                            solution->upper = upperZ;
                            solution->sinAngle = sinAngle;
                            solution->cosAngle = cosAngle;
                        }

                        solIdx = atomicAdd(&nSK2CSolutions, 1);

                        if (solIdx < MAX_SK_PHASE_TWO_C) {
                            struct SKPhase2* solution = &(sk2CSolutions[solIdx]);
                            solution->p1Idx = idx;
                            solution->f2Angle = (f2Angle + 32768) % 65536;
                            solution->tenKFloorIdx = i;
                            solution->lower = lowerZ;
                            solution->upper = upperZ;
                            solution->sinAngle = -sinAngle;
                            solution->cosAngle = -cosAngle;
                        }
                    }
                }
                else {
                    float lowerX = INFINITY;
                    float upperX = -INFINITY;

                    double tanAngle = sinAngle / cosAngle;

                    for (int i = 0; i < nPoints; i++) {
                        float testX = floorPoints[3 * i] + ((midPointZ - floorPoints[3 * i + 2]) * tanAngle) - midPointX;
                        lowerX = fminf(lowerX, testX);
                        upperX = fmaxf(upperX, testX);
                    }

                    for (int i = 0; i < nTenKFloors; i++) {
                        int solIdx = atomicAdd(&nSK2DSolutions, 1);

                        if (solIdx < MAX_SK_PHASE_TWO_D) {
                            struct SKPhase2* solution = &(sk2DSolutions[solIdx]);
                            solution->p1Idx = idx;
                            solution->f2Angle = f2Angle;
                            solution->tenKFloorIdx = i;
                            solution->lower = lowerX;
                            solution->upper = upperX;
                            solution->sinAngle = sinAngle;
                            solution->cosAngle = cosAngle;
                        }

                        solIdx = atomicAdd(&nSK2DSolutions, 1);

                        if (solIdx < MAX_SK_PHASE_TWO_D) {
                            struct SKPhase2* solution = &(sk2DSolutions[solIdx]);
                            solution->p1Idx = idx;
                            solution->f2Angle = (f2Angle + 32768) % 65536;
                            solution->tenKFloorIdx = i;
                            solution->lower = lowerX;
                            solution->upper = upperX;
                            solution->sinAngle = -sinAngle;
                            solution->cosAngle = -cosAngle;
                        }
                    }
                }
            }
        }
    }
}

__global__ void find_slide_kick_setupG(short* floorPoints, const int nPoints, float floorNormalY, double maxSpeed, int maxF1PU, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int x1 = 4 * (idx % (2 * (maxF1PU / 4) + 1)) - maxF1PU;
    int z1 = 4 * (idx / (2 * (maxF1PU / 4) + 1)) - maxF1PU;

    if ((x1 != 0 || z1 != 0) && 65536.0 * sqrt((double)(x1 * x1 + z1 * z1)) <= floorNormalY * maxSpeed) {
        float dx = 65536 * x1;
        float dy = 500.0f;
        float dz = 65536 * z1;

        float d = sqrtf(dx * dx + dy * dy + dz * dz);

        d = 1.0 / d;
        dx *= d;
        dy *= d;
        dz *= d;

        float normal_change[3];
        normal_change[0] = (platformNormal[0] <= dx) ? ((dx - platformNormal[0] < 0.01f) ? dx - platformNormal[0] : 0.01f) : ((dx - platformNormal[0] > -0.01f) ? dx - platformNormal[0] : -0.01f);
        normal_change[1] = (platformNormal[1] <= dy) ? ((dy - platformNormal[1] < 0.01f) ? dy - platformNormal[1] : 0.01f) : ((dy - platformNormal[1] > -0.01f) ? dy - platformNormal[1] : -0.01f);
        normal_change[2] = (platformNormal[2] <= dz) ? ((dz - platformNormal[2] < 0.01f) ? dz - platformNormal[2] : 0.01f) : ((dz - platformNormal[2] > -0.01f) ? dz - platformNormal[2] : -0.01f);

        if (normal_change[0] == normal_offsets[t][0] && normal_change[1] == normal_offsets[t][1] && normal_change[2] == normal_offsets[t][2]) {
            double qStepMul = (nPoints == 4) ? 1.0 : (4.0 / 3.0);

            double maxF1Dist = -INFINITY;
            double minF1Dist = INFINITY;

            int refAngle = 0;

            int maxF1Angle = -65536;
            int minF1Angle = 65536;

            for (int i = 0; i < nPoints; i++) {
                for (int j = 0; j < nPoints; j++) {
                    double xDist = (65536.0 * x1 + (floorPoints[3 * i] - floorPoints[3 * j]) * qStepMul);
                    double zDist = (65536.0 * z1 + (floorPoints[3 * i + 2] - floorPoints[3 * j + 2]) * qStepMul);

                    double dist = sqrt(xDist * xDist + zDist * zDist);

                    minF1Dist = fmin(minF1Dist, dist);
                    maxF1Dist = fmax(maxF1Dist, dist);

                    int angle = atan2sG(zDist, xDist);

                    if (i == 0 && j == 0) {
                        refAngle = angle;
                    }

                    angle = (short)(angle - refAngle);

                    minF1Angle = min(minF1Angle, angle);
                    maxF1Angle = max(maxF1Angle, angle);

                }
            }

            double minSpeedF1 = minF1Dist / floorNormalY;
            double maxSpeedF1 = fmin(maxSpeed, maxF1Dist / floorNormalY);

            if (minSpeedF1 < maxSpeedF1) {
                minF1Angle = (65536 + minF1Angle + refAngle) % 65536;
                maxF1Angle = (65536 + maxF1Angle + refAngle) % 65536;

                int minF1AngleIdx = gReverseArctanTableG[minF1Angle];
                int maxF1AngleIdx = gReverseArctanTableG[maxF1Angle];

                if (maxF1AngleIdx < minF1AngleIdx) {
                    maxF1AngleIdx = maxF1AngleIdx + 8192;
                }

                for (int q2 = 1; q2 <= 4; q2++) {
                    int solIdx = atomicAdd(&nSK1Solutions, 1);

                    if (solIdx < MAX_SK_PHASE_ONE) {
                        struct SKPhase1* solution = &(sk1Solutions[solIdx]);
                        solution->x1 = x1;
                        solution->z1 = z1;
                        solution->q2 = q2;
                        solution->minSpeed = minSpeedF1;
                        solution->maxSpeed = maxSpeedF1;
                        solution->minF1Dist = minF1Dist;
                        solution->maxF1Dist = maxF1Dist;
                        solution->minF1AngleIdx = minF1AngleIdx;
                        solution->maxF1AngleIdx = maxF1AngleIdx;
                    }
                }
            }
        }
    }
}

void find_slide_kick_setup_triangle(short* floorPoints, short* devFloorPoints, int nPoints, float yNormal, int t, double maxSpeed, int nThreads) {
    int nSK1SolutionsCPU = 0;
    int nSK2ASolutionsCPU = 0;
    int nSK2BSolutionsCPU = 0;
    int nSK2CSolutionsCPU = 0;
    int nSK2DSolutionsCPU = 0;
    int nSK3SolutionsCPU = 0;
    int nSK4SolutionsCPU = 0;
    int nSK5SolutionsCPU = 0;
    int nSK6SolutionsCPU = 0;

    cudaMemcpyToSymbol(nSK1Solutions, &nSK1SolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nSK2ASolutions, &nSK2ASolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nSK2BSolutions, &nSK2BSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nSK2CSolutions, &nSK2CSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nSK2DSolutions, &nSK2DSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nSK3Solutions, &nSK3SolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nSK4Solutions, &nSK4SolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nSK5Solutions, &nSK5SolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nSK6Solutions, &nSK6SolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

    float platformMinX = 32767.0;
    float platformMaxX = -32768.0;
    float platformMinZ = 32767.0;
    float platformMaxZ = -32768.0;

    for (int i = 0; i < nPoints; i++) {
        platformMinX = fminf(platformMinX, (float)floorPoints[3 * i]);
        platformMaxX = fmaxf(platformMaxX, (float)floorPoints[3 * i]);
    }

    for (int i = 0; i < nPoints; i++) {
        platformMinZ = fminf(platformMinZ, (float)floorPoints[3 * i + 2]);
        platformMaxZ = fmaxf(platformMaxZ, (float)floorPoints[3 * i + 2]);
    }

    float midPointX = 0.0f;
    float midPointZ = 0.0f;

    for (int i = 0; i < nPoints; i++) {
        midPointX += floorPoints[3 * i];
        midPointZ += floorPoints[3 * i + 2];
    }

    midPointX /= (float)nPoints;
    midPointZ /= (float)nPoints;

    cudaMemcpy(devFloorPoints, floorPoints, 3 * nPoints * sizeof(short), cudaMemcpyHostToDevice);

    int maxF1PU = (int)floor(yNormal * maxSpeed / (4.0 * 65536.0)) * 4;
    int nBlocks = ((2 * (maxF1PU / 4) + 1) * (2 * (maxF1PU / 4) + 1) + nThreads - 1) / nThreads;

    find_slide_kick_setupG<<<nBlocks, nThreads>>>(devFloorPoints, nPoints, yNormal, maxSpeed, maxF1PU, t);

    cudaMemcpyFromSymbol(&nSK1SolutionsCPU, nSK1Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);

    if (nSK1SolutionsCPU > 0) {
        if (nSK1SolutionsCPU > MAX_SK_PHASE_ONE) {
            fprintf(stderr, "Warning: Number of phase 1 solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nSK1SolutionsCPU = MAX_SK_PHASE_ONE;
        }

        nBlocks = (nSK1SolutionsCPU + nThreads - 1) / nThreads;

        find_slide_kick_setupG2<<<nBlocks, nThreads>>>(devFloorPoints, nPoints, yNormal, platformMinX, platformMaxX, platformMinZ, platformMaxZ, midPointX, midPointZ);

        cudaMemcpyFromSymbol(&nSK2ASolutionsCPU, nSK2ASolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&nSK2BSolutionsCPU, nSK2BSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&nSK2CSolutionsCPU, nSK2CSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&nSK2DSolutionsCPU, nSK2DSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    }

    if (nSK2ASolutionsCPU > 0) {
        if (nSK2ASolutionsCPU > MAX_SK_PHASE_TWO_A) {
            fprintf(stderr, "Warning: Number of phase 2a solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nSK2ASolutionsCPU = MAX_SK_PHASE_TWO_A;
        }

        nBlocks = (nSK2ASolutionsCPU + nThreads - 1) / nThreads;

        find_slide_kick_setupG3a<<<nBlocks, nThreads>>>(platformMinZ, platformMaxZ);
    }

    if (nSK2BSolutionsCPU > 0) {
        if (nSK2BSolutionsCPU > MAX_SK_PHASE_TWO_B) {
            fprintf(stderr, "Warning: Number of phase 2b solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nSK2BSolutionsCPU = MAX_SK_PHASE_TWO_B;
        }

        nBlocks = (nSK2BSolutionsCPU + nThreads - 1) / nThreads;

        find_slide_kick_setupG3b<<<nBlocks, nThreads>>>(platformMinX, platformMaxX);
    }

    if (nSK2CSolutionsCPU > 0) {
        if (nSK2CSolutionsCPU > MAX_SK_PHASE_TWO_C) {
            fprintf(stderr, "Warning: Number of phase 2c solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nSK2CSolutionsCPU = MAX_SK_PHASE_TWO_C;
        }

        nBlocks = (nSK2CSolutionsCPU + nThreads - 1) / nThreads;

        find_slide_kick_setupG3c<<<nBlocks, nThreads >>>(platformMinX, platformMaxX, platformMinZ, platformMaxZ);
    }

    if (nSK2DSolutionsCPU > 0) {
        if (nSK2DSolutionsCPU > MAX_SK_PHASE_TWO_D) {
            fprintf(stderr, "Warning: Number of phase 2d solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nSK2DSolutionsCPU = MAX_SK_PHASE_TWO_D;
        }

        nBlocks = (nSK2DSolutionsCPU + nThreads - 1) / nThreads;

        find_slide_kick_setupG3d<<<nBlocks, nThreads>>>(platformMinX, platformMaxX, platformMinZ, platformMaxZ);
    }

    cudaMemcpyFromSymbol(&nSK3SolutionsCPU, nSK3Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);

    if (nSK3SolutionsCPU > 0) {
        if (nSK3SolutionsCPU > MAX_SK_PHASE_THREE) {
            fprintf(stderr, "Warning: Number of phase 3 solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nSK3SolutionsCPU = MAX_SK_PHASE_THREE;
        }

        nBlocks = (nSK3SolutionsCPU + nThreads - 1) / nThreads;

        try_slide_kick_routeG<<<nBlocks, nThreads>>>(devFloorPoints, nPoints);

        cudaMemcpyFromSymbol(&nSK4SolutionsCPU, nSK4Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);

    }

    if (nSK4SolutionsCPU > 0) {
        if (nSK4SolutionsCPU > MAX_SK_PHASE_FOUR) {
            fprintf(stderr, "Warning: Number of phase 4 solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nSK4SolutionsCPU = MAX_SK_PHASE_FOUR;
        }

        nBlocks = (nSK4SolutionsCPU + nThreads - 1) / nThreads;

        try_slide_kick_routeG2<<<nBlocks, nThreads>>>();

        cudaMemcpyFromSymbol(&nSK5SolutionsCPU, nSK5Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    }

    if (nSK5SolutionsCPU > 0) {
        if (nSK5SolutionsCPU > MAX_SK_PHASE_FIVE) {
            fprintf(stderr, "Warning: Number of phase 5 solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nSK5SolutionsCPU = MAX_SK_PHASE_FIVE;
        }

        nBlocks = (nSK5SolutionsCPU + nThreads - 1) / nThreads;

        try_stick_positionG<<<nBlocks, nThreads>>>();
    }
}

__global__ void find_bully_positions(int uphillAngle, float maxSlidingSpeed, float maxSlidingSpeedToPlatform) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int squishPushFrames = (idx % 3) + 2;
    idx = idx / 3;

    if (idx < min(nDouble10KSolutions, MAX_DOUBLE_10K_SOLUTIONS)) {
        struct DoubleTenKSolution* doubleTenKSol = &(doubleTenKSolutions[idx]);
        struct TenKSolution* tenKSol = &(tenKSolutions[doubleTenKSol->tenKSolutionIdx]);

        const float accel = 7.0f;
        const float pushRadius = 115.0f;
        const float bullyHurtbox = 63.0f;
        const float baseBullySpeed = powf(2.0f, 24);
        const float maxBullySpeed = nextafterf(powf(2.0f, 30), -INFINITY);

        int floorIdx = (tenKSol->squishCeiling == 0 || tenKSol->squishCeiling == 2) ? 0 : 1;

        int surfAngle = atan2sG(squishCeilingNormals[tenKSol->squishCeiling][2], squishCeilingNormals[tenKSol->squishCeiling][0]);
        surfAngle = (65536 + surfAngle) % 65536;

        float xPushVel = gSineTableG[surfAngle >> 4] * 10.0f;
        float zPushVel = gCosineTableG[surfAngle >> 4] * 10.0f;

        int slopeAngle = atan2sG(startNormals[floorIdx][2], startNormals[floorIdx][0]);
        slopeAngle = (slopeAngle + 65536) % 65536;

        float steepness = sqrtf(startNormals[floorIdx][0] * startNormals[floorIdx][0] + startNormals[floorIdx][2] * startNormals[floorIdx][2]);

        float slopeXVel = accel * steepness * gSineTableG[slopeAngle >> 4];
        float slopeZVel = accel * steepness * gCosineTableG[slopeAngle >> 4];

        float minBullyPushX = doubleTenKSol->minStartX;
        float maxBullyPushX = doubleTenKSol->maxStartX;
        float minBullyPushZ = doubleTenKSol->minStartZ;
        float maxBullyPushZ = doubleTenKSol->maxStartZ;

        for (int i = 0; i < squishPushFrames-1; i++) {
            minBullyPushX = minBullyPushX - squishNormals[floorIdx][1] * xPushVel / 4.0f;
            maxBullyPushX = maxBullyPushX - squishNormals[floorIdx][1] * xPushVel / 4.0f;
            minBullyPushZ = minBullyPushZ - squishNormals[floorIdx][1] * zPushVel / 4.0f;
            maxBullyPushZ = maxBullyPushZ - squishNormals[floorIdx][1] * zPushVel / 4.0f;
        }

        minBullyPushX = minBullyPushX - xPushVel / 4.0f;
        maxBullyPushX = maxBullyPushX - xPushVel / 4.0f;
        minBullyPushZ = minBullyPushZ - zPushVel / 4.0f;
        maxBullyPushZ = maxBullyPushZ - zPushVel / 4.0f;

        int minAngle = INT_MAX;
        int maxAngle = INT_MIN;
        int refAngle = 65536;

        for (int j = 0; j < 4; j++) {
            float bullyPushX = (j % 2 == 0) ? minBullyPushX : maxBullyPushX;
            float bullyPushZ = (j / 2 == 0) ? minBullyPushZ : maxBullyPushZ;

            for (int k = 0; k < nSquishSpots[tenKSol->squishCeiling]; k++) {
                float signX = (squishSpots[(2 * tenKSol->squishCeiling * MAX_SQUISH_SPOTS) + (2 * k)] > 0) - (squishSpots[(2 * tenKSol->squishCeiling * MAX_SQUISH_SPOTS) + (2 * k)] < 0);
                float signZ = (squishSpots[(2 * k) + 1] > 0) - (squishSpots[(2 * k) + 1] < 0);

                for (int l = 0; l < 4; l++) {
                    float xDist = bullyPushX - (squishSpots[(2 * tenKSol->squishCeiling * MAX_SQUISH_SPOTS) + (2 * k)] + signX * (l % 2));
                    float zDist = bullyPushZ - (squishSpots[(2 * tenKSol->squishCeiling * MAX_SQUISH_SPOTS) + (2 * k) + 1] + signZ * (l / 2));

                    float dist = sqrtf(xDist * xDist + zDist * zDist);

                    if (dist >= pushRadius - bullyHurtbox && dist <= pushRadius - fmaxf(bullyHurtbox - 2.0f * maxSlidingSpeed - 1.85f, 0.0f)) {
                        int angle = atan2sG(zDist, xDist);
                        angle = (angle + 65536) % 65536;

                        int angleDiff = (short)(angle - uphillAngle);

                        if (angleDiff < -0x4000 || angleDiff > 0x4000) {
                            if (refAngle == 65536) {
                                refAngle = angle;
                            }

                            minAngle = min(minAngle, (int)(short)(angle - refAngle));
                            maxAngle = max(maxAngle, (int)(short)(angle - refAngle));
                        }
                    }
                }
            }
        }

        if (refAngle != 65536) {
            minAngle = (minAngle + refAngle + 65536) % 65536;
            maxAngle = (maxAngle + refAngle + 65536) % 65536;

            int minAngleIdx = gReverseArctanTableG[minAngle];
            int maxAngleIdx = gReverseArctanTableG[maxAngle];

            while (((((gArctanTableG[minAngleIdx] + 65536) % 65536) >> 4) << 4) < minAngle) {
                minAngleIdx = (minAngleIdx + 1) % 8192;
            }

            if (maxAngleIdx < minAngleIdx) {
                maxAngleIdx = maxAngleIdx + 8192;
            }

            for (int j = minAngleIdx; j <= maxAngleIdx; j++) {
                int angle = (gArctanTableG[j % 8192] + 65536) % 65536;

                float minBullyX = minBullyPushX - pushRadius * gSineTableG[angle >> 4];
                float maxBullyX = maxBullyPushX - pushRadius * gSineTableG[angle >> 4];
                float minBullyZ = minBullyPushZ - pushRadius * gCosineTableG[angle >> 4];
                float maxBullyZ = maxBullyPushZ - pushRadius * gCosineTableG[angle >> 4];

                float xDiff2;

                if (minBullyX == maxBullyX) {
                    int precision;
                    frexpf(minBullyX, &precision);
                    xDiff2 = powf(2.0f, precision - 24);
                }
                else {
                    xDiff2 = powf(2.0f, floorf(log2f(maxBullyX - minBullyX)));

                    while (floorf(maxBullyX / (2.0f * xDiff2)) >= ceilf(minBullyX / (2.0f * xDiff2))) {
                        xDiff2 = xDiff2 * 2.0f;
                    }
                }

                float zDiff2;

                if (minBullyZ == maxBullyZ) {
                    int precision;
                    frexpf(minBullyZ, &precision);
                    zDiff2 = powf(2.0f, precision - 24);
                }
                else {
                    zDiff2 = powf(2.0f, floorf(log2f(maxBullyZ - minBullyZ)));

                    while (floorf(maxBullyZ / (2.0f * zDiff2)) >= ceilf(minBullyZ / (2.0f * zDiff2))) {
                        zDiff2 = zDiff2 * 2.0f;
                    }
                }

                float maxBullyXSpeed = fminf(nextafterf(xDiff2 * baseBullySpeed, -INFINITY), maxBullySpeed);
                float maxBullyZSpeed = fminf(nextafterf(zDiff2 * baseBullySpeed, -INFINITY), maxBullySpeed);

                float maxPushSpeed = (fabsf(maxBullyXSpeed * gSineTableG[angle >> 4]) + fabsf(maxBullyZSpeed * gCosineTableG[angle >> 4])) * (73.0f / 53.0f) * 3.0f;

                float maxLossFactor = (-1.0 * (0.5f + 0.5f * maxPushSpeed / 100.0f)) * 0.02 + 0.92;
                float slidingSpeedX = (doubleTenKSol->post10KXVel / maxLossFactor) - slopeXVel;
                float slidingSpeedZ = (doubleTenKSol->post10KZVel / maxLossFactor) - slopeZVel;

                float slidingSpeedToPlatformOptions[4] = { -slidingSpeedX, slidingSpeedZ, -slidingSpeedZ, slidingSpeedX };

                float slidingSpeedToPlatform = slidingSpeedToPlatformOptions[tenKSol->squishCeiling];

                if (fabsf(slidingSpeedX) <= maxSlidingSpeed && fabsf(slidingSpeedZ) <= maxSlidingSpeed && slidingSpeedToPlatform <= maxSlidingSpeedToPlatform) {
                    int solIdx = atomicAdd(&nBullyPushSolutions, 1);

                    if (solIdx < MAX_BULLY_PUSH_SOLUTIONS) {
                        struct BullyPushSolution* solution = &(bullyPushSolutions[solIdx]);
                        solution->doubleTenKSolutionIdx = idx;
                        solution->bullyMinX = minBullyX;
                        solution->bullyMaxX = maxBullyX;
                        solution->bullyMinZ = minBullyZ;
                        solution->bullyMaxZ = maxBullyZ;
                        solution->pushAngle = angle;
                        solution->maxSpeed = maxPushSpeed;
                        solution->squishPushQF = squishPushFrames;
                        solution->squishPushMinX = minBullyPushX;
                        solution->squishPushMaxX = maxBullyPushX;
                        solution->squishPushMinZ = minBullyPushZ;
                        solution->squishPushMaxZ = maxBullyPushZ;
                        solution->minSlidingSpeedX = slidingSpeedX;
                        solution->minSlidingSpeedZ = slidingSpeedZ;
                        atomicAdd(&(tenKSol->bpSetups), 1);
                    }
                }
            }
        }
    }
}

__device__ float find_speed_boundary(float minValue, float maxValue, float target, float yNormal, float vel, int dir) {
    while (nextafterf(minValue, INFINITY) < maxValue) {
        float midValue = fmaxf(nextafterf(minValue, INFINITY), (minValue + maxValue) / 2.0f);

        float pos = midValue;

        for (int i = 0; i < 4; i++) {
            pos = pos + yNormal * (vel / 4.0f);
        }

        if (pos < target) {
            minValue = midValue;
        }
        else if (pos > target) {
            maxValue = midValue;
        }
        else if (dir > 0) {
            minValue = midValue;
        }
        else {
            maxValue = midValue;
        }
    }

    if (dir > 0) {
        return minValue;
    }
    else {
        return maxValue;
    }
}

__device__ bool search_xVel(float& xVel, float zVel, float targetSpeed, float* targetPosition, float yNormal, int idx) {
    float xDir = (xVel > 0) - (xVel < 0);

    float speed = sqrtf(xVel * xVel + zVel * zVel);

    float minXVel = xVel;
    float maxXVel = xVel;

    float minSpeed = speed;

    if (xDir * minSpeed < xDir * targetSpeed) {
        while (xDir * minSpeed < xDir * targetSpeed) {
            minXVel = nextafterf(minXVel, INFINITY);
            minSpeed = sqrtf(minXVel * minXVel + zVel * zVel);
        }
    }
    else {
        while (xDir * minSpeed >= xDir * targetSpeed) {
            minXVel = nextafterf(minXVel, -INFINITY);
            minSpeed = sqrtf(minXVel * minXVel + zVel * zVel);
        }

        minXVel = nextafterf(minXVel, INFINITY);
    }

    float maxSpeed = speed;

    if (xDir * maxSpeed > xDir * targetSpeed) {
        while (xDir * maxSpeed > xDir * targetSpeed) {
            maxXVel = nextafterf(maxXVel, -INFINITY);
            maxSpeed = sqrtf(maxXVel * maxXVel + zVel * zVel);
        }
    }
    else {
        while (xDir * maxSpeed <= xDir * targetSpeed) {
            maxXVel = nextafterf(maxXVel, INFINITY);
            maxSpeed = sqrtf(maxXVel * maxXVel + zVel * zVel);
        }

        maxXVel = nextafterf(maxXVel, -INFINITY);
    }

    bool foundSpeed = minXVel > maxXVel;

    for (float xVel1 = minXVel; xVel1 <= maxXVel; xVel1 = nextafterf(xVel1, INFINITY)) {
        float minX = nextafterf(targetPosition[0], -INFINITY) - yNormal * xVel1;
        float maxX = nextafterf(targetPosition[0], INFINITY) - yNormal * xVel1;
        float minZ = nextafterf(targetPosition[2], -INFINITY) - yNormal * zVel;
        float maxZ = nextafterf(targetPosition[2], INFINITY) - yNormal * zVel;

        minX = find_speed_boundary(minX, maxX, targetPosition[0], yNormal, xVel1, -1);
        maxX = find_speed_boundary(minX, maxX, targetPosition[0], yNormal, xVel1, 1);
        minZ = find_speed_boundary(minZ, maxZ, targetPosition[2], yNormal, zVel, -1);
        maxZ = find_speed_boundary(minZ, maxZ, targetPosition[2], yNormal, zVel, 1);

        int minXI = (int)minX;
        int maxXI = (int)maxX;
        int minZI = (int)minZ;
        int maxZI = (int)maxZ;

        float minXF = INFINITY;
        float maxXF = -INFINITY;
        float minZF = INFINITY;
        float maxZF = -INFINITY;

        for (int x = minXI; x <= maxXI; x++) {
            for (int z = minZI; z <= maxZI; z++) {
                float squarePos[3] = {(float)x, 0.0f, (float)z};
                float fHeight;
                int fIdx1 = find_floor(squarePos, squishTriangles, squishNormals, &fHeight);
                int fIdx2 = find_floor(squarePos, startTriangles, startNormals, &fHeight);

                if (fIdx1 == -1 && fIdx2 != -1) {
                    minXF = fminf(minXF, (x < 0) ? nextafterf((float)(x - 1), INFINITY) : (float)x);
                    maxXF = fmaxf(maxXF, (x > 0) ? nextafterf((float)(x + 1), -INFINITY) : (float)x);
                    minZF = fminf(minZF, (z < 0) ? nextafterf((float)(z - 1), INFINITY) : (float)z);
                    maxZF = fmaxf(maxZF, (z > 0) ? nextafterf((float)(z + 1), -INFINITY) : (float)z);
                }
            }
        }

        minX = fmaxf(minX, minXF);
        maxX = fminf(maxX, maxXF);
        minZ = fmaxf(minZ, minZF);
        maxZ = fminf(maxZ, maxZF);

        if (minX <= maxX && minZ <= maxZ) {
            int solIdx = atomicAdd(&nDouble10KSolutions, 1);

            if (solIdx < MAX_DOUBLE_10K_SOLUTIONS) {
                struct DoubleTenKSolution* solution = &(doubleTenKSolutions[solIdx]);
                solution->tenKSolutionIdx = idx;
                solution->post10KXVel = xVel1;
                solution->post10KZVel = zVel;
                solution->minStartX = minX;
                solution->maxStartX = maxX;
                solution->minStartZ = minZ;
                solution->maxStartZ = maxZ;
            }

            xVel = xVel1;
        }
    }

    return foundSpeed;
}

__global__ void find_double_10k_solutions() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(n10KSolutions, MAX_10K_SOLUTIONS)) {
        struct TenKSolution* tenKSol = &(tenKSolutions[idx]);

        if (tenKSol->bdSetups > 0) {
            struct SpeedSolution* speedSol = &(speedSolutions[tenKSol->speedSolutionIdx]);

            int floorIdx = (tenKSol->squishCeiling == 0 || tenKSol->squishCeiling == 2) ? 0 : 1;

            double xPos = (tenKSol->startPosition[0][0] + tenKSol->startPosition[1][0]) / 2.0;
            double zPos = (tenKSol->startPosition[0][2] + tenKSol->startPosition[1][2]) / 2.0;

            double xDiff = tenKSol->frame1Position[0] - xPos;
            double zDiff = tenKSol->frame1Position[2] - zPos;

            float xVel = xDiff / startNormals[floorIdx][1];
            float zVel = zDiff / startNormals[floorIdx][1];

            bool searchLoop = true;
            float xVel1 = xVel;

            for (float zVel1 = zVel; searchLoop && abs(zVel1) <= tenKSol->departureSpeed; zVel1 = nextafterf(zVel1, -INFINITY)) {
                searchLoop = search_xVel(xVel1, zVel1, tenKSol->departureSpeed, tenKSol->frame1Position, startNormals[floorIdx][1], idx);
            }

            searchLoop = true;
            xVel1 = xVel;

            for (float zVel1 = nextafterf(zVel, INFINITY); searchLoop && abs(zVel1) <= tenKSol->departureSpeed; zVel1 = nextafterf(zVel1, INFINITY)) {
                searchLoop = search_xVel(xVel1, zVel1, tenKSol->departureSpeed, tenKSol->frame1Position, startNormals[floorIdx][1], idx);
            }
        }
    }
}

__global__ void set_squish_spots(short* tris, float* norms) {
    for (int x = 0; x < 4; x++) {
        for (int y = 0; y < 3; y++) {
            preSquishCeilingTriangles[x][y][0] = tris[9 * x + 3 * y];
            preSquishCeilingTriangles[x][y][1] = tris[9 * x + 3 * y + 1];
            preSquishCeilingTriangles[x][y][2] = tris[9 * x + 3 * y + 2];
            preSquishCeilingNormals[x][y] = norms[3 * x + y];

            squishCeilingTriangles[x][y][0] = tris[36 + 9 * x + 3 * y];
            squishCeilingTriangles[x][y][1] = tris[36 + 9 * x + 3 * y + 1];
            squishCeilingTriangles[x][y][2] = tris[36 + 9 * x + 3 * y + 2];
            squishCeilingNormals[x][y] = norms[12 + 3 * x + y];

            startCeilingTriangles[x][y][0] = tris[72 + 9 * x + 3 * y];
            startCeilingTriangles[x][y][1] = tris[72 + 9 * x + 3 * y + 1];
            startCeilingTriangles[x][y][2] = tris[72 + 9 * x + 3 * y + 2];
            startCeilingNormals[x][y] = norms[24 + 3 * x + y];
        }

        squishCeilings[x] = squishCeilingNormals[x][1] > -0.5f;
    }

    for (int ceilIdx = 0; ceilIdx < 4; ceilIdx++) {
        nSquishSpots[ceilIdx] = 0;

        if (squishCeilings[ceilIdx]) {
            int minX = INT_MAX;
            int maxX = INT_MIN;
            int minZ = INT_MAX;
            int maxZ = INT_MIN;

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 3; j++) {
                    minX = min(minX, (int)squishCeilingTriangles[i][j][0]);
                    maxX = max(maxX, (int)squishCeilingTriangles[i][j][0]);
                    minZ = min(minZ, (int)squishCeilingTriangles[i][j][2]);
                    maxZ = max(maxZ, (int)squishCeilingTriangles[i][j][2]);
                }
            }

            int x0 = (ceilIdx == 0 || ceilIdx == 2) ? squishCeilingTriangles[ceilIdx][0][0] : squishCeilingTriangles[ceilIdx][1][0];
            int z0 = (ceilIdx == 0 || ceilIdx == 2) ? squishCeilingTriangles[ceilIdx][0][2] : squishCeilingTriangles[ceilIdx][1][2];
            int x1 = ceilIdx == 0 ? squishCeilingTriangles[ceilIdx][1][0] : squishCeilingTriangles[ceilIdx][2][0];
            int z1 = ceilIdx == 0 ? squishCeilingTriangles[ceilIdx][1][2] : squishCeilingTriangles[ceilIdx][2][2];

            float xDiff = x1 - x0;
            float zDiff = z1 - z0;

            float diffDist = sqrtf(xDiff * xDiff + zDiff * zDiff);

            xDiff = xDiff / diffDist;
            zDiff = zDiff / diffDist;

            for (int x = minX; x <= maxX; x++) {
                for (int z = minZ; z <= maxZ; z++) {
                    float pos[3] = { x, -3071.0f , z };

                    float ceilHeight;
                    int idx = find_ceil(pos, squishCeilingTriangles, squishCeilingNormals, &ceilHeight);

                    if (idx != -1 && idx == ceilIdx && ceilHeight - -3071.0f < 150.0f) {
                        float floorHeight;
                        int floorIdx = find_floor(pos, squishTriangles, squishNormals, &floorHeight);

                        if (floorIdx == -1) {
                            float preCeilHeight;
                            int preIdx = find_ceil(pos, preSquishCeilingTriangles, preSquishCeilingNormals, &preCeilHeight);

                            if (preIdx == -1) {
                                if (nSquishSpots[ceilIdx] == MAX_SQUISH_SPOTS) {
                                    printf("Warning: Number of squish spots for this normal has been exceeded. No more squish spots for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                                }

                                if (nSquishSpots[ceilIdx] < MAX_SQUISH_SPOTS) {
                                    squishSpots[(2 * ceilIdx * MAX_SQUISH_SPOTS) + (2 * nSquishSpots[ceilIdx])] = x;
                                    squishSpots[(2 * ceilIdx * MAX_SQUISH_SPOTS) + (2 * nSquishSpots[ceilIdx]) + 1] = z;
                                }

                                nSquishSpots[ceilIdx]++;
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void init_squish_spots(float* devSquishSpots, int* devNSquishSpots) {
    squishSpots = devSquishSpots;
    nSquishSpots = devNSquishSpots;
}

__global__ void generate_strain_setups() {
    nStrainSetups = 0;

    for (int i = 0; i <= maxFSpeedLevels; i++) {
        float fSpeed = (float)i / (float)maxFSpeedLevels;

        float maxSSpeedLevelsSq = maxSSpeedLevels * sqrtf(1.0 - fSpeed * fSpeed);

        fSpeed = 1.5f * fSpeed;

        for (int j = 0; j <= maxSSpeedLevelsSq; j++) {
            float sSpeed = 10.0f * (float)j / (float)maxSSpeedLevels;

            for (float signF = (i == 0) ? 1.0f : -1.0f; signF <= 1.0f; signF += 2.0f) {
                for (float signS = (j == 0) ? 1.0f : -1.0f; signS <= 1.0f; signS += 2.0f) {
                    if (nStrainSetups < MAX_STRAIN_SETUPS) {
                        struct StrainSetup* setup = &(strainSetups[nStrainSetups]);
                        setup->forwardStrain = signF*fSpeed;
                        setup->sidewardStrain = signS*sSpeed;
                    }

                    nStrainSetups++;
                }
            }
        }
    }
}

__global__ void init_strain_setups(StrainSetup* devStrainSetups) {
    strainSetups = devStrainSetups;
}

__global__ void reset_ranges() {
    maxFSpeedLevels = 0;
    maxSSpeedLevels = 0;
}

__global__ void copy_solution_pointers(SolStruct s) {
    sk1Solutions = s.sk1Solutions;
    sk2ASolutions = s.sk2ASolutions;
    sk2BSolutions = s.sk2BSolutions;
    sk2CSolutions = s.sk2CSolutions;
    sk2DSolutions = s.sk2DSolutions;
    sk3Solutions = s.sk3Solutions;
    sk4Solutions = s.sk4Solutions;
    sk5Solutions = s.sk5Solutions;
    sk6Solutions = s.sk6Solutions;
    platSolutions = s.platSolutions;
    upwarpSolutions = s.upwarpSolutions;
    skuwSolutions = s.skuwSolutions;
    speedSolutions = s.speedSolutions;
    tenKSolutions = s.tenKSolutions;
    doubleTenKSolutions = s.doubleTenKSolutions;
    bullyPushSolutions = s.bullyPushSolutions;
    slideSolutions = s.slideSolutions;
    bdSolutions = s.bdSolutions;
}

int init_solution_structs(SolStruct* s) {
    int errorCode = 0;

    errorCode |= cudaMalloc((void**)&s->sk1Solutions, MAX_SK_PHASE_ONE * sizeof(SKPhase1));
    errorCode |= cudaMalloc((void**)&s->sk2ASolutions, MAX_SK_PHASE_TWO_A * sizeof(SKPhase2));
    errorCode |= cudaMalloc((void**)&s->sk2BSolutions, MAX_SK_PHASE_TWO_B * sizeof(SKPhase2));
    errorCode |= cudaMalloc((void**)&s->sk2CSolutions, MAX_SK_PHASE_TWO_C * sizeof(SKPhase2));
    errorCode |= cudaMalloc((void**)&s->sk2DSolutions, MAX_SK_PHASE_TWO_D * sizeof(SKPhase2));
    errorCode |= cudaMalloc((void**)&s->sk3Solutions, MAX_SK_PHASE_THREE * sizeof(SKPhase3));
    errorCode |= cudaMalloc((void**)&s->sk4Solutions, MAX_SK_PHASE_FOUR * sizeof(SKPhase4));
    errorCode |= cudaMalloc((void**)&s->sk5Solutions, MAX_SK_PHASE_FIVE * sizeof(SKPhase5));
    errorCode |= cudaMalloc((void**)&s->sk6Solutions, MAX_SK_PHASE_SIX * sizeof(SKPhase6));
    errorCode |= cudaMalloc((void**)&s->platSolutions, MAX_PLAT_SOLUTIONS * sizeof(PlatformSolution));
    errorCode |= cudaMalloc((void**)&s->upwarpSolutions, MAX_UPWARP_SOLUTIONS * sizeof(UpwarpSolution));
    errorCode |= cudaMalloc((void**)&s->skuwSolutions, MAX_SK_UPWARP_SOLUTIONS * sizeof(SKUpwarpSolution));
    errorCode |= cudaMalloc((void**)&s->speedSolutions, MAX_SPEED_SOLUTIONS * sizeof(SpeedSolution));
    errorCode |= cudaMalloc((void**)&s->tenKSolutions, MAX_10K_SOLUTIONS * sizeof(TenKSolution));
    errorCode |= cudaMalloc((void**)&s->doubleTenKSolutions, MAX_DOUBLE_10K_SOLUTIONS * sizeof(DoubleTenKSolution));
    errorCode |= cudaMalloc((void**)&s->bullyPushSolutions, MAX_BULLY_PUSH_SOLUTIONS * sizeof(BullyPushSolution));
    errorCode |= cudaMalloc((void**)&s->slideSolutions, MAX_SLIDE_SOLUTIONS * sizeof(SlideSolution));
    errorCode |= cudaMalloc((void**)&s->bdSolutions, MAX_BD_SOLUTIONS * sizeof(BDSolution));

    copy_solution_pointers<<<1, 1>>>(*s);

    return errorCode;
}

void free_solution_pointers(SolStruct* s) {
    cudaFree(s->sk1Solutions);
    cudaFree(s->sk2ASolutions);
    cudaFree(s->sk2BSolutions);
    cudaFree(s->sk2CSolutions);
    cudaFree(s->sk2DSolutions);
    cudaFree(s->sk3Solutions);
    cudaFree(s->sk4Solutions);
    cudaFree(s->sk5Solutions);
    cudaFree(s->sk6Solutions);
    cudaFree(s->platSolutions);
    cudaFree(s->upwarpSolutions);
    cudaFree(s->skuwSolutions);
    cudaFree(s->speedSolutions);
    cudaFree(s->tenKSolutions);
    cudaFree(s->doubleTenKSolutions);
    cudaFree(s->bullyPushSolutions);
    cudaFree(s->slideSolutions);
    cudaFree(s->bdSolutions);

}


void write_solutions_to_file(Vec3f startNormal, struct FSTOptions* o, struct FSTData* p, int floorIdx, std::ofstream& wf) {
    int nSK1SolutionsCPU = 0;
    int nSK2ASolutionsCPU = 0;
    int nSK2BSolutionsCPU = 0;
    int nSK2CSolutionsCPU = 0;
    int nSK2DSolutionsCPU = 0;
    int nSK3SolutionsCPU = 0;
    int nSK4SolutionsCPU = 0;
    int nSK5SolutionsCPU = 0;
    int nSK6SolutionsCPU = 0;
    int nPlatSolutionsCPU = 0;
    int nUpwarpSolutionsCPU = 0;
    int nSKUWSolutionsCPU = 0;
    int nSpeedSolutionsCPU = 0;
    int n10KSolutionsCPU = 0;
    int nSlideSolutionsCPU = 0;
    int nBDSolutionsCPU = 0;
    int nDouble10KSolutionsCPU = 0;
    int nBullyPushSolutionsCPU = 0;

    cudaMemcpyFromSymbol(&nSK1SolutionsCPU, nSK1Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nSK2ASolutionsCPU, nSK2ASolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nSK2BSolutionsCPU, nSK2BSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nSK2CSolutionsCPU, nSK2CSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nSK2DSolutionsCPU, nSK2DSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nSK3SolutionsCPU, nSK3Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nSK4SolutionsCPU, nSK4Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nSK5SolutionsCPU, nSK5Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nSK6SolutionsCPU, nSK6Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nPlatSolutionsCPU, nPlatSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nUpwarpSolutionsCPU, nUpwarpSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nSKUWSolutionsCPU, nSKUWSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nSpeedSolutionsCPU, nSpeedSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&n10KSolutionsCPU, n10KSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nSlideSolutionsCPU, nSlideSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nBDSolutionsCPU, nBDSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nDouble10KSolutionsCPU, nDouble10KSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nBullyPushSolutionsCPU, nBullyPushSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);

    nSK1SolutionsCPU = min(nSK1SolutionsCPU, MAX_SK_PHASE_ONE);
    nSK2ASolutionsCPU = min(nSK2ASolutionsCPU, MAX_SK_PHASE_TWO_A);
    nSK2BSolutionsCPU = min(nSK2BSolutionsCPU, MAX_SK_PHASE_TWO_B);
    nSK2CSolutionsCPU = min(nSK2CSolutionsCPU, MAX_SK_PHASE_TWO_C);
    nSK2DSolutionsCPU = min(nSK2DSolutionsCPU, MAX_SK_PHASE_TWO_D);
    nSK3SolutionsCPU = min(nSK3SolutionsCPU, MAX_SK_PHASE_THREE);
    nSK4SolutionsCPU = min(nSK4SolutionsCPU, MAX_SK_PHASE_FOUR);
    nSK5SolutionsCPU = min(nSK5SolutionsCPU, MAX_SK_PHASE_FIVE);
    nSK6SolutionsCPU = min(nSK6SolutionsCPU, MAX_SK_PHASE_SIX);
    nPlatSolutionsCPU = min(nPlatSolutionsCPU, MAX_PLAT_SOLUTIONS);
    nUpwarpSolutionsCPU = min(nUpwarpSolutionsCPU, MAX_UPWARP_SOLUTIONS);
    nSKUWSolutionsCPU = min(nSKUWSolutionsCPU, MAX_SK_UPWARP_SOLUTIONS);
    nSpeedSolutionsCPU = min(nSpeedSolutionsCPU, MAX_SPEED_SOLUTIONS);
    n10KSolutionsCPU = min(n10KSolutionsCPU, MAX_10K_SOLUTIONS);
    nSlideSolutionsCPU = min(nSlideSolutionsCPU, MAX_SLIDE_SOLUTIONS);
    nBDSolutionsCPU = min(nBDSolutionsCPU, MAX_BD_SOLUTIONS);
    nDouble10KSolutionsCPU = min(nDouble10KSolutionsCPU, MAX_DOUBLE_10K_SOLUTIONS);
    nBullyPushSolutionsCPU = min(nBullyPushSolutionsCPU, MAX_BULLY_PUSH_SOLUTIONS);

    struct DoubleTenKSolution* doubleTenKSolutionsCPU = (struct DoubleTenKSolution*)std::malloc(nDouble10KSolutionsCPU * sizeof(struct DoubleTenKSolution));
    struct BullyPushSolution* bullyPushSolutionsCPU = (struct BullyPushSolution*)std::malloc(nBullyPushSolutionsCPU * sizeof(struct BullyPushSolution));

    struct BDSolution* bdSolutionsCPU = (struct BDSolution*)std::malloc(nBDSolutionsCPU * sizeof(struct BDSolution));
    struct SlideSolution* slideSolutionsCPU = (struct SlideSolution*)std::malloc(nSlideSolutionsCPU * sizeof(struct SlideSolution));
    struct TenKSolution* tenKSolutionsCPU = (struct TenKSolution*)std::malloc(n10KSolutionsCPU * sizeof(struct TenKSolution));
    struct SpeedSolution* speedSolutionsCPU = (struct SpeedSolution*)std::malloc(nSpeedSolutionsCPU * sizeof(struct SpeedSolution));
    struct SKUpwarpSolution* skuwSolutionsCPU = (struct SKUpwarpSolution*)std::malloc(nSKUWSolutionsCPU * sizeof(struct SKUpwarpSolution));

    struct PlatformSolution* platSolutionsCPU = (struct PlatformSolution*)std::malloc(nPlatSolutionsCPU * sizeof(struct PlatformSolution));
    struct UpwarpSolution* upwarpSolutionsCPU = (struct UpwarpSolution*)std::malloc(nUpwarpSolutionsCPU * sizeof(struct UpwarpSolution));

    struct SKPhase1* sk1SolutionsCPU = (struct SKPhase1*)std::malloc(nSK1SolutionsCPU * sizeof(struct SKPhase1));
    struct SKPhase2* sk2ASolutionsCPU = (struct SKPhase2*)std::malloc(nSK2ASolutionsCPU * sizeof(struct SKPhase2));
    struct SKPhase2* sk2BSolutionsCPU = (struct SKPhase2*)std::malloc(nSK2BSolutionsCPU * sizeof(struct SKPhase2));
    struct SKPhase2* sk2CSolutionsCPU = (struct SKPhase2*)std::malloc(nSK2CSolutionsCPU * sizeof(struct SKPhase2));
    struct SKPhase2* sk2DSolutionsCPU = (struct SKPhase2*)std::malloc(nSK2DSolutionsCPU * sizeof(struct SKPhase2));
    struct SKPhase3* sk3SolutionsCPU = (struct SKPhase3*)std::malloc(nSK3SolutionsCPU * sizeof(struct SKPhase3));
    struct SKPhase4* sk4SolutionsCPU = (struct SKPhase4*)std::malloc(nSK4SolutionsCPU * sizeof(struct SKPhase4));
    struct SKPhase5* sk5SolutionsCPU = (struct SKPhase5*)std::malloc(nSK5SolutionsCPU * sizeof(struct SKPhase5));
    struct SKPhase6* sk6SolutionsCPU = (struct SKPhase6*)std::malloc(nSK6SolutionsCPU * sizeof(struct SKPhase6));

    cudaMemcpy(doubleTenKSolutionsCPU, p->s.doubleTenKSolutions, nDouble10KSolutionsCPU * sizeof(struct DoubleTenKSolution), cudaMemcpyDeviceToHost);
    cudaMemcpy(bullyPushSolutionsCPU, p->s.bullyPushSolutions, nBullyPushSolutionsCPU * sizeof(struct BullyPushSolution), cudaMemcpyDeviceToHost);

    cudaMemcpy(bdSolutionsCPU, p->s.bdSolutions, nBDSolutionsCPU * sizeof(struct BDSolution), cudaMemcpyDeviceToHost);
    cudaMemcpy(slideSolutionsCPU, p->s.slideSolutions, nSlideSolutionsCPU * sizeof(struct SlideSolution), cudaMemcpyDeviceToHost);
    cudaMemcpy(tenKSolutionsCPU, p->s.tenKSolutions, n10KSolutionsCPU * sizeof(struct TenKSolution), cudaMemcpyDeviceToHost);
    cudaMemcpy(speedSolutionsCPU, p->s.speedSolutions, nSpeedSolutionsCPU * sizeof(struct SpeedSolution), cudaMemcpyDeviceToHost);
    cudaMemcpy(skuwSolutionsCPU, p->s.skuwSolutions, nSKUWSolutionsCPU * sizeof(struct SKUpwarpSolution), cudaMemcpyDeviceToHost);

    cudaMemcpy(upwarpSolutionsCPU, p->s.upwarpSolutions, nUpwarpSolutionsCPU * sizeof(struct UpwarpSolution), cudaMemcpyDeviceToHost);
    cudaMemcpy(platSolutionsCPU, p->s.platSolutions, nPlatSolutionsCPU * sizeof(struct PlatformSolution), cudaMemcpyDeviceToHost);

    cudaMemcpy(sk1SolutionsCPU, p->s.sk1Solutions, nSK1SolutionsCPU * sizeof(struct SKPhase1), cudaMemcpyDeviceToHost);
    cudaMemcpy(sk2ASolutionsCPU, p->s.sk2ASolutions, nSK2ASolutionsCPU * sizeof(struct SKPhase2), cudaMemcpyDeviceToHost);
    cudaMemcpy(sk2BSolutionsCPU, p->s.sk2BSolutions, nSK2BSolutionsCPU * sizeof(struct SKPhase2), cudaMemcpyDeviceToHost);
    cudaMemcpy(sk2CSolutionsCPU, p->s.sk2CSolutions, nSK2CSolutionsCPU * sizeof(struct SKPhase2), cudaMemcpyDeviceToHost);
    cudaMemcpy(sk2DSolutionsCPU, p->s.sk2DSolutions, nSK2DSolutionsCPU * sizeof(struct SKPhase2), cudaMemcpyDeviceToHost);
    cudaMemcpy(sk3SolutionsCPU, p->s.sk3Solutions, nSK3SolutionsCPU * sizeof(struct SKPhase3), cudaMemcpyDeviceToHost);
    cudaMemcpy(sk4SolutionsCPU, p->s.sk4Solutions, nSK4SolutionsCPU * sizeof(struct SKPhase4), cudaMemcpyDeviceToHost);
    cudaMemcpy(sk5SolutionsCPU, p->s.sk5Solutions, nSK5SolutionsCPU * sizeof(struct SKPhase5), cudaMemcpyDeviceToHost);
    cudaMemcpy(sk6SolutionsCPU, p->s.sk6Solutions, nSK6SolutionsCPU * sizeof(struct SKPhase6), cudaMemcpyDeviceToHost);

    int bdRunningSum = 0;
    int bpRunningSum = 0;

    int solTotal = 0;

    for (int l = 0; l < n10KSolutionsCPU; l++) {
        solTotal += tenKSolutionsCPU[l].bdSetups * tenKSolutionsCPU[l].bpSetups;
        bdRunningSum += tenKSolutionsCPU[l].bdSetups;
        tenKSolutionsCPU[l].bdSetups = bdRunningSum - tenKSolutionsCPU[l].bdSetups;
        bpRunningSum += tenKSolutionsCPU[l].bpSetups;
        tenKSolutionsCPU[l].bpSetups = bpRunningSum - tenKSolutionsCPU[l].bpSetups;
    }

    printf("        # Full Solutions: %d\n", solTotal);

    if (o->minimalOutput) {
        if (solTotal > 0) {
            wf << startNormal[0] << "," << startNormal[1] << "," << startNormal[2] << endl;
        }
    }
    else {
        for (int l = 0; l < nBDSolutionsCPU; l++) {
            if (bdSolutionsCPU[l].slideSolutionIdx >= 0) {
                int tenKIdx = slideSolutionsCPU[bdSolutionsCPU[l].slideSolutionIdx].tenKSolutionIdx;

                int newIdx = tenKSolutionsCPU[tenKIdx].bdSetups;
                tenKSolutionsCPU[tenKIdx].bdSetups++;

                if (l != newIdx) {
                    struct BDSolution temp = bdSolutionsCPU[l];
                    temp.slideSolutionIdx = -temp.slideSolutionIdx - 1;
                    bdSolutionsCPU[l] = bdSolutionsCPU[newIdx];
                    bdSolutionsCPU[newIdx] = temp;
                    l--;
                }
            }
            else {
                bdSolutionsCPU[l].slideSolutionIdx = -bdSolutionsCPU[l].slideSolutionIdx - 1;
            }
        }

        for (int l = 0; l < nBullyPushSolutionsCPU; l++) {
            if (bullyPushSolutionsCPU[l].doubleTenKSolutionIdx >= 0) {
                int tenKIdx = doubleTenKSolutionsCPU[bullyPushSolutionsCPU[l].doubleTenKSolutionIdx].tenKSolutionIdx;

                int newIdx = tenKSolutionsCPU[tenKIdx].bpSetups;
                tenKSolutionsCPU[tenKIdx].bpSetups++;

                if (l != newIdx) {
                    struct BullyPushSolution temp = bullyPushSolutionsCPU[l];
                    temp.doubleTenKSolutionIdx = -temp.doubleTenKSolutionIdx - 1;
                    bullyPushSolutionsCPU[l] = bullyPushSolutionsCPU[newIdx];
                    bullyPushSolutionsCPU[newIdx] = temp;
                    l--;
                }
            }
            else {
                bullyPushSolutionsCPU[l].doubleTenKSolutionIdx = -bullyPushSolutionsCPU[l].doubleTenKSolutionIdx - 1;
            }
        }

        int m = 0;

        for (int l = 0; l < nBDSolutionsCPU; l++) {
            struct BDSolution* bdSol = &(bdSolutionsCPU[l]);
            struct SlideSolution* slideSol = &(slideSolutionsCPU[bdSol->slideSolutionIdx]);
            struct TenKSolution* tenKSol = &(tenKSolutionsCPU[slideSol->tenKSolutionIdx]);
            struct SpeedSolution* speedSol = &(speedSolutionsCPU[tenKSol->speedSolutionIdx]);
            struct SKUpwarpSolution* skuwSol = &(skuwSolutionsCPU[speedSol->skuwSolutionIdx]);
            struct UpwarpSolution* uwSol = &(upwarpSolutionsCPU[skuwSol->uwIdx]);
            struct PlatformSolution* platSol = &(platSolutionsCPU[uwSol->platformSolutionIdx]);
            struct SKPhase6* p6Sol = &(sk6SolutionsCPU[skuwSol->skIdx]);
            struct SKPhase5* p5Sol = &(sk5SolutionsCPU[p6Sol->p5Idx]);
            struct SKPhase4* p4Sol = &(sk4SolutionsCPU[p5Sol->p4Idx]);
            struct SKPhase3* p3Sol = &(sk3SolutionsCPU[p4Sol->p3Idx]);
            struct SKPhase2* p2Sol = (p3Sol->p2Type / 2 == 0) ? ((p3Sol->p2Type % 2 == 0) ? &(sk2ASolutionsCPU[p3Sol->p2Idx]) : &(sk2BSolutionsCPU[p3Sol->p2Idx])) : ((p3Sol->p2Type % 2 == 0) ? &(sk2CSolutionsCPU[p3Sol->p2Idx]) : &(sk2DSolutionsCPU[p3Sol->p2Idx]));
            struct SKPhase1* p1Sol = &(sk1SolutionsCPU[p2Sol->p1Idx]);

            while (m < nBullyPushSolutionsCPU && doubleTenKSolutionsCPU[bullyPushSolutionsCPU[m].doubleTenKSolutionIdx].tenKSolutionIdx < slideSol->tenKSolutionIdx) {
                m++;
            }

            while (m < nBullyPushSolutionsCPU && doubleTenKSolutionsCPU[bullyPushSolutionsCPU[m].doubleTenKSolutionIdx].tenKSolutionIdx == slideSol->tenKSolutionIdx) {
                struct BullyPushSolution* bpSol = &(bullyPushSolutionsCPU[m]);
                struct DoubleTenKSolution* doubleTenKSol = &(doubleTenKSolutionsCPU[bpSol->doubleTenKSolutionIdx]);

                //printf("---------------------------------------\nFound Solution:\n---------------------------------------\n    Start Position Range: [%.10g, %.10g], [%.10g, %.10g]\n    Frame 1 Position: %.10g, %.10g, %.10g\n    Frame 2 Position: %.10g, %.10g, %.10g\n    Return Position: %.10g, %.10g, %.10g\n    PU Departure Speed: %.10g (x=%.10g, z=%.10g)\n    PU Strain Speed: (x=%.10g, z=%.10g, fwd=%.10g)\n    Pre-10K Speed: (x=%.10g, z=%.10g)\n    PU Return Speed: %.10g (x=%.10g, z=%.10g)\n    Frame 1 Q-steps: %d\n    Frame 2 Q-steps: %d\n    Frame 3 Q-steps: %d\n", doubleTenKSol->minStartX, doubleTenKSol->maxStartX, doubleTenKSol->minStartZ, doubleTenKSol->maxStartZ, tenKSol->frame1Position[0], tenKSol->frame1Position[1], tenKSol->frame1Position[2], tenKSol->frame2Position[0], tenKSol->frame2Position[1], tenKSol->frame2Position[2], platSol->returnPosition[0], platSol->returnPosition[1], platSol->returnPosition[2], tenKSol->departureSpeed, doubleTenKSol->post10KXVel, doubleTenKSol->post10KZVel, speedSol->xStrain, speedSol->zStrain, speedSol->forwardStrain, tenKSol->pre10KVel[0], tenKSol->pre10KVel[1], speedSol->returnSpeed, tenKSol->returnVel[0], tenKSol->returnVel[1], 4, p1Sol->q2, 1);
                //printf("    10k Stick X: %d\n    10k Stick Y: %d\n    Frame 2 HAU: %d\n    10k Camera Yaw: %d\n    Start Floor Normal: %.10g, %.10g, %.10g\n", ((p5Sol->stickX == 0) ? 0 : ((p5Sol->stickX < 0) ? p5Sol->stickX - 6 : p5Sol->stickX + 6)), ((p5Sol->stickY == 0) ? 0 : ((p5Sol->stickY < 0) ? p5Sol->stickY - 6 : p5Sol->stickY + 6)), p2Sol->f2Angle, p4Sol->cameraYaw, host_norms[3 * x], host_norms[3 * x + 1], host_norms[3 * x + 2]);
                //printf("---------------------------------------\n    Tilt Frames: %d\n    Post-Tilt Platform Normal: %.10g, %.10g, %.10g\n    Post-Tilt Position: %.10g, %.10g, %.10g\n    Pre-Upwarp Position: %.10g, %.10g, %.10g\n    Post-Upwarp Position: %.10g, %.10g, %.10g\n    Upwarp PU X: %d\n    Upwarp PU Z: %d\n    Upwarp Slide Facing Angle: %d\n    Upwarp Slide Intended Mag: %.10g\n    Upwarp Slide Intended DYaw: %d\n", platSol->nFrames, platSol->endNormal[0], platSol->endNormal[1], platSol->endNormal[2], platSol->endPosition[0], platSol->endPosition[1], platSol->endPosition[2], slideSol->preUpwarpPosition[0], slideSol->preUpwarpPosition[1], slideSol->preUpwarpPosition[2], slideSol->upwarpPosition[0], slideSol->upwarpPosition[1], slideSol->upwarpPosition[2], uwSol->pux, uwSol->puz, slideSol->angle, slideSol->stickMag, slideSol->intendedDYaw);
                //printf("---------------------------------------\n    Post-Breakdance Camera Yaw: %d\n    Post-Breakdance Stick X: %d\n    Post-Breakdance Stick Y: %d\n    Landing Position: %.10g, %.10g, %.10g\n    Landing Speed: %.10g\n---------------------------------------\n\n\n", bdSol->cameraYaw, bdSol->stickX, bdSol->stickY, bdSol->landingPosition[0], bdSol->landingPosition[1], bdSol->landingPosition[2], bdSol->postSlideSpeed);
                //printf("---------------------------------------\n    Squish Push Position Range: [%.10g, %.10g], [%.10g, %.10g]\n    Squish Push Q-steps: %d\n    Bully Position Range: [%.10g, %.10g], [%.10g, %.10g]\n    Bully Push Angle: %d\n    Max Bully Speed: %.10g\n    Min Sliding Spped: (x=%.10g, z=%.10g)\n---------------------------------------\n\n\n", bpSol->squishPushMinX, bpSol->squishPushMaxX, bpSol->squishPushMinZ, bpSol->squishPushMaxZ, bpSol->squishPushQF, bpSol->bullyMinX, bpSol->bullyMaxX, bpSol->bullyMinZ, bpSol->bullyMaxZ, bpSol->pushAngle, bpSol->maxSpeed, bpSol->minSlidingSpeedX, bpSol->minSlidingSpeedZ);

                wf << startNormal[0] << "," << startNormal[1] << "," << startNormal[2] << ",";
                wf << doubleTenKSol->minStartX << "," << doubleTenKSol->maxStartX << ",";
                wf << doubleTenKSol->minStartZ << "," << doubleTenKSol->maxStartZ << ",";
                wf << tenKSol->frame1Position[0] << "," << tenKSol->frame1Position[1] << "," << tenKSol->frame1Position[2] << ",";
                wf << tenKSol->frame2Position[0] << "," << tenKSol->frame2Position[1] << "," << tenKSol->frame2Position[2] << ",";
                wf << platSol->returnPosition[0] << "," << platSol->returnPosition[1] << "," << platSol->returnPosition[2] << ",";
                wf << tenKSol->departureSpeed << "," << doubleTenKSol->post10KXVel << "," << doubleTenKSol->post10KZVel << ",";
                wf << speedSol->xStrain << "," << speedSol->zStrain << "," << speedSol->forwardStrain << ",";
                wf << tenKSol->pre10KVel[0] << "," << tenKSol->pre10KVel[1] << ",";
                wf << speedSol->returnSpeed << "," << tenKSol->returnVel[0] << "," << tenKSol->returnVel[1] << ",";
                wf << 4 << "," << p1Sol->q2 << "," << 1 << ",";
                wf << ((p5Sol->stickX == 0) ? 0 : ((p5Sol->stickX < 0) ? p5Sol->stickX - 6 : p5Sol->stickX + 6)) << "," << ((p5Sol->stickY == 0) ? 0 : ((p5Sol->stickY < 0) ? p5Sol->stickY - 6 : p5Sol->stickY + 6)) << ",";
                wf << p2Sol->f2Angle << "," << p4Sol->cameraYaw << ",";
                wf << p->host_norms[3 * floorIdx] << "," << p->host_norms[3 * floorIdx + 1] << "," << p->host_norms[3 * floorIdx + 2] << ",";
                wf << platSol->nFrames << ",";
                wf << platSol->endNormal[0] << "," << platSol->endNormal[1] << "," << platSol->endNormal[2] << ",";
                wf << platSol->endPosition[0] << "," << platSol->endPosition[1] << "," << platSol->endPosition[2] << ",";
                wf << slideSol->preUpwarpPosition[0] << "," << slideSol->preUpwarpPosition[1] << "," << slideSol->preUpwarpPosition[2] << ",";
                wf << slideSol->upwarpPosition[0] << "," << slideSol->upwarpPosition[1] << "," << slideSol->upwarpPosition[2] << ",";
                wf << uwSol->pux << "," << uwSol->puz << ",";
                wf << slideSol->angle << "," << slideSol->stickMag << "," << slideSol->intendedDYaw << ",";
                wf << bdSol->cameraYaw << ",";
                wf << bdSol->stickX << "," << bdSol->stickY << ",";
                wf << bdSol->landingPosition[0] << "," << bdSol->landingPosition[1] << "," << bdSol->landingPosition[2] << ",";
                wf << bdSol->postSlideSpeed << ",";
                wf << bpSol->squishPushMinX << "," << bpSol->squishPushMaxX << "," << bpSol->squishPushMinZ << "," << bpSol->squishPushMaxZ << "," << bpSol->squishPushQF << ",";
                wf << bpSol->bullyMinX << "," << bpSol->bullyMaxX << "," << bpSol->bullyMinZ << "," << bpSol->bullyMaxZ << "," << bpSol->pushAngle << ",";
                wf << bpSol->maxSpeed << "," << bpSol->minSlidingSpeedX << "," << bpSol->minSlidingSpeedZ << endl;

                m++;
            }
        }
    }

    std::free(bullyPushSolutionsCPU);
    std::free(doubleTenKSolutionsCPU);
    std::free(bdSolutionsCPU);
    std::free(slideSolutionsCPU);
    std::free(tenKSolutionsCPU);
    std::free(speedSolutionsCPU);
    std::free(skuwSolutionsCPU);
    std::free(upwarpSolutionsCPU);
    std::free(platSolutionsCPU);
    std::free(sk1SolutionsCPU);
    std::free(sk2ASolutionsCPU);
    std::free(sk2BSolutionsCPU);
    std::free(sk2CSolutionsCPU);
    std::free(sk2DSolutionsCPU);
    std::free(sk3SolutionsCPU);
    std::free(sk4SolutionsCPU);
    std::free(sk5SolutionsCPU);
    std::free(sk6SolutionsCPU);
}

void write_solution_file_header(bool minimalOutput, std::ofstream& wf) {
    wf << "Start Normal X,Start Normal Y,Start Normal Z";

    if (!minimalOutput) {
        wf << ",";
        wf << "Start Position Min X,Start Position Max X,";
        wf << "Start Position Min Z,Start Position Max Z,";
        wf << "Frame 1 Position X,Frame 1 Position Y,Frame 1 Position Z,";
        wf << "Frame 2 Position X,Frame 2 Position Y,Frame 2 Position Z,";
        wf << "Return Position X,Return Position Y,Return Position Z,";
        wf << "Departure Speed,Departure X Velocity,Departure Z Velocity,";
        wf << "Frame 2 Strain X Velocity,Frame 2 Strain Z Velocity,Frame 2 Strain Forward Speed,";
        wf << "Pre-10K X Velocity, Pre-10K Z Velocity,";
        wf << "Return Speed,Return X Velocity,Return Z Velocity,";
        wf << "Frame 1 Q-steps,Frame 2 Q-steps,Frame 3 Q-steps,";
        wf << "10K Stick X,10K Stick Y,";
        wf << "Frame 2 HAU,10K Camera Yaw,";
        wf << "Start Floor Normal X,Start Floor Normal Y,Start Floor Normal Z,";
        wf << "Number of Tilt Frames,";
        wf << "Post-Tilt Platform Normal X,Post-Tilt Platform Normal Y,Post-Tilt Platform Normal Z,";
        wf << "Post-Tilt Position X,Post-Tilt Position Y,Post-Tilt Position Z,";
        wf << "Pre-Upwarp Position X,Pre-Upwarp Position Y,Pre-Upwarp Position Z,";
        wf << "Post-Upwarp Position X,Post-Upwarp Position Y,Post-Upwarp Position Z,";
        wf << "Upwarp PU X,Upwarp PU Z,";
        wf << "Upwarp Slide Facing Angle,Upwarp Slide IntendedMag,Upwarp Slide IntendedDYaw,";
        wf << "Post-Breakdance Camera Yaw,";
        wf << "Post-Breakdance Stick X,Post-Breakdance Stick Y,";
        wf << "Landing Position X,Landing Position Y,Landing Position Z,";
        wf << "Landing Speed,";
        wf << "Squish Push Min X,Squish Push Max X,Squish Push Min Z,Squish Push Max Z,Squish Push Q-steps,";
        wf << "Bully Min X,Bully Max X,Bully Min Z,Bully Max Z,Bully Push HAU,";
        wf << "Max Bully Push Speed,Min X Sliding Speed,Min Z Sliding Speed";
    }

    wf << endl;
}

bool check_normal(Vec3f startNormal, struct FSTOptions* o, struct FSTData* p, std::ofstream& wf) {
    bool foundSolution = false;

    const float normal_offsets_cpu[4][3] = { {0.01f, -0.01f, 0.01f}, {-0.01f, -0.01f, 0.01f}, {-0.01f, -0.01f, -0.01f}, {0.01f, -0.01f, -0.01f} };

    Vec3f preStartNormal = { (startNormal[0] + (startNormal[0] > 0 ? 0.01f : -0.01f)) + (startNormal[0] > 0 ? 0.01f : -0.01f) , (startNormal[1] - 0.01f) - 0.01f, (startNormal[2] + (startNormal[2] > 0 ? 0.01f : -0.01f)) + (startNormal[2] > 0 ? 0.01f : -0.01f) };
    Vec3f offPlatformPosition = { o->platformPos[0], 2000.0f, o->platformPos[2] };

    Platform platform0 = Platform(o->platformPos[0], o->platformPos[1], o->platformPos[2], preStartNormal);
    platform0.platform_logic(offPlatformPosition);

    Platform platform1 = Platform(o->platformPos[0], o->platformPos[1], o->platformPos[2], platform0.normal);
    platform1.platform_logic(offPlatformPosition);

    startNormal[0] = platform1.normal[0];
    startNormal[1] = platform1.normal[1];
    startNormal[2] = platform1.normal[2];

    set_platform_normal<<<1, 1>>>(startNormal[0], startNormal[1], startNormal[2]);
    set_platform_pos<<<1, 1>>>(o->platformPos[0], o->platformPos[1], o->platformPos[2]);

    float ceilingNormals[4] = { platform1.ceilings[0].normal[1], platform1.ceilings[1].normal[1], platform1.ceilings[2].normal[1], platform1.ceilings[3].normal[1] };
    bool squishTest = (ceilingNormals[0] > -0.5f) || (ceilingNormals[1] > -0.5f) || (ceilingNormals[2] > -0.5f) || (ceilingNormals[3] > -0.5f);

    if (!squishTest) {
        return false;
    }

    int uphillAngle = atan2s(-platform1.normal[2], -platform1.normal[0]);
    uphillAngle = (65536 + uphillAngle) % 65536;

    Platform platform = Platform(o->platformPos[0], o->platformPos[1], o->platformPos[2], startNormal);
    platform.platform_logic(offPlatformPosition);

    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 3; y++) {
            p->host_tris[9 * x + 3 * y] = platform.triangles[x].vectors[y][0];
            p->host_tris[9 * x + 3 * y + 1] = platform.triangles[x].vectors[y][1];
            p->host_tris[9 * x + 3 * y + 2] = platform.triangles[x].vectors[y][2];
            p->host_norms[3 * x + y] = platform.triangles[x].normal[y];
            p->host_tris[18 + 9 * x + 3 * y] = platform1.triangles[x].vectors[y][0];
            p->host_tris[18 + 9 * x + 3 * y + 1] = platform1.triangles[x].vectors[y][1];
            p->host_tris[18 + 9 * x + 3 * y + 2] = platform1.triangles[x].vectors[y][2];
            p->host_norms[6 + 3 * x + y] = platform1.triangles[x].normal[y];
        }
    }
                    
    cudaMemcpy(p->dev_tris, p->host_tris, 36 * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(p->dev_norms, p->host_norms, 12 * sizeof(float), cudaMemcpyHostToDevice);

    set_start_triangle<<<1, 1>>>(p->dev_tris, p->dev_norms);

    for (int x = 0; x < 4; x++) {
        for (int y = 0; y < 3; y++) {
            p->host_ceiling_tris[9 * x + 3 * y] = platform0.ceilings[x].vectors[y][0];
            p->host_ceiling_tris[9 * x + 3 * y + 1] = platform0.ceilings[x].vectors[y][1];
            p->host_ceiling_tris[9 * x + 3 * y + 2] = platform0.ceilings[x].vectors[y][2];
            p->host_ceiling_norms[3 * x + y] = platform0.ceilings[x].normal[y];

            p->host_ceiling_tris[36 + 9 * x + 3 * y] = platform1.ceilings[x].vectors[y][0];
            p->host_ceiling_tris[36 + 9 * x + 3 * y + 1] = platform1.ceilings[x].vectors[y][1];
            p->host_ceiling_tris[36 + 9 * x + 3 * y + 2] = platform1.ceilings[x].vectors[y][2];
            p->host_ceiling_norms[12 + 3 * x + y] = platform1.ceilings[x].normal[y];

            p->host_ceiling_tris[72 + 9 * x + 3 * y] = platform.ceilings[x].vectors[y][0];
            p->host_ceiling_tris[72 + 9 * x + 3 * y + 1] = platform.ceilings[x].vectors[y][1];
            p->host_ceiling_tris[72 + 9 * x + 3 * y + 2] = platform.ceilings[x].vectors[y][2];
            p->host_ceiling_norms[24 + 3 * x + y] = platform.ceilings[x].normal[y];
        }
    }

    cudaMemcpy(p->dev_ceiling_tris, p->host_ceiling_tris, 108 * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(p->dev_ceiling_norms, p->host_ceiling_norms, 36 * sizeof(float), cudaMemcpyHostToDevice);

    set_squish_spots<<<1, 1>>>(p->dev_ceiling_tris, p->dev_ceiling_norms);

    Vec3f postTiltNormal = { platform.normal[0], platform.normal[1], platform.normal[2] };
                    
    for (int t = 0; t < 4; t++) {
        platform.normal[0] = postTiltNormal[0] + normal_offsets_cpu[t][0];
        platform.normal[1] = postTiltNormal[1] + normal_offsets_cpu[t][1];
        platform.normal[2] = postTiltNormal[2] + normal_offsets_cpu[t][2];

        for (int k = 2; k < o->nPUFrames; k++) {
            platform.platform_logic(offPlatformPosition);
        }

        float minX = INT16_MAX;
        float maxX = INT16_MIN;
        float minZ = INT16_MAX;
        float maxZ = INT16_MIN;

        for (int k = 0; k < platform.triangles.size(); k++) {
            minX = fminf(fminf(fminf(minX, platform.triangles[k].vectors[0][0]), platform.triangles[k].vectors[1][0]), platform.triangles[k].vectors[2][0]);
            maxX = fmaxf(fmaxf(fmaxf(maxX, platform.triangles[k].vectors[0][0]), platform.triangles[k].vectors[1][0]), platform.triangles[k].vectors[2][0]);
            minZ = fminf(fminf(fminf(minZ, platform.triangles[k].vectors[0][2]), platform.triangles[k].vectors[1][2]), platform.triangles[k].vectors[2][2]);
            maxZ = fmaxf(fmaxf(fmaxf(maxZ, platform.triangles[k].vectors[0][2]), platform.triangles[k].vectors[1][2]), platform.triangles[k].vectors[2][2]);
        }
                        
        int nX = round((maxX - minX) / o->deltaX) + 1;
        int nZ = round((maxZ - minZ) / o->deltaZ) + 1;

        int nPlatSolutionsCPU = 0;
        int nUpwarpSolutionsCPU = 0;

        cudaMemcpyToSymbol(nPlatSolutions, &nPlatSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

        long long int nBlocks = (nX * nZ + o->nThreads - 1) / o->nThreads;

        cudaFunc<<<nBlocks, o->nThreads>>>(minX, o->deltaX, minZ, o->deltaZ, nX, nZ, platform.normal[0], platform.normal[1], platform.normal[2], o->maxFrames);

        cudaMemcpyFromSymbol(&nPlatSolutionsCPU, nPlatSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                        
        if (nPlatSolutionsCPU > 0) {
            if (nPlatSolutionsCPU > MAX_PLAT_SOLUTIONS) {
                fprintf(stderr, "Warning: Number of platform solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                nPlatSolutionsCPU = MAX_PLAT_SOLUTIONS;
            }

            //printf("---------------------------------------\nTesting Normal: %.10g, %.10g, %.10g\nPlatform Position: %.10g, %.10g, %.10g\n        Index: %d, %d, %d, %d\n", startNormal[0], startNormal[1], startNormal[2], platformPos[0], platformPos[1], platformPos[2], h, i, j, quad);
            //printf("        # Platform Solutions: %d\n", nPlatSolutionsCPU);

            nBlocks = (nPlatSolutionsCPU + o->nThreads - 1) / o->nThreads;

            cudaMemcpyToSymbol(nUpwarpSolutions, &nUpwarpSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

            find_upwarp_solutions<<<nBlocks, o->nThreads>>>(1000000000.0f);

            cudaMemcpyFromSymbol(&nUpwarpSolutionsCPU, nUpwarpSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
        }

        if (nUpwarpSolutionsCPU > 0) {
            if (nUpwarpSolutionsCPU > MAX_UPWARP_SOLUTIONS) {
                fprintf(stderr, "Warning: Number of upwarp solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                nUpwarpSolutionsCPU = MAX_UPWARP_SOLUTIONS;
            }

            //printf("        # Upwarp Solutions: %d\n", nUpwarpSolutionsCPU);

            bool sameNormal = p->host_norms[1] == p->host_norms[4];

            for (int x = 0; x < (sameNormal ? 1 : 2); x++) {
                for (int y = 0; y < 3; y++) {
                    p->floorPoints[3 * y] = p->host_tris[9 * x + 3 * y];
                    p->floorPoints[3 * y + 1] = p->host_tris[9 * x + 3 * y + 1];
                    p->floorPoints[3 * y + 2] = p->host_tris[9 * x + 3 * y + 2];
                }

                if (sameNormal) {
                    p->floorPoints[9] = p->host_tris[15];
                    p->floorPoints[10] = p->host_tris[16];
                    p->floorPoints[11] = p->host_tris[17];
                }

                p->squishEdges[0] = (x == 0) ? 2 : -1;
                p->squishEdges[1] = (x == 0) ? 0 : 1;
                p->squishEdges[2] = (sameNormal || x == 1) ? (sameNormal ? 1 : 3) : -1;
                p->squishEdges[3] = sameNormal ? 3 : -1;

                for (int y = 0; y < 4; y++) {
                    if (p->squishEdges[y] != -1 && ceilingNormals[p->squishEdges[y]] <= -0.5f) {
                        p->squishEdges[y] = -1;
                    }
                }

                cudaMemcpy(p->devSquishEdges, p->squishEdges, 4 * sizeof(int), cudaMemcpyHostToDevice);

                int nSK6SolutionsCPU = 0;
                int nSKUWSolutionsCPU = 0;
                int nSpeedSolutionsCPU = 0;
                int n10KSolutionsCPU = 0;
                int nDouble10KSolutionsCPU = 0;
                int nBullyPushSolutionsCPU = 0;
                int nSlideSolutionsCPU = 0;
                int nBDSolutionsCPU = 0;

                int nStrainSetupsCPU = 0;

                find_slide_kick_setup_triangle(p->floorPoints, p->devFloorPoints, sameNormal ? 4 : 3, p->host_norms[3 * x + 1], t, o->maxSpeed, o->nThreads);

                cudaMemcpyFromSymbol(&nSK6SolutionsCPU, nSK6Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);

                if (nSK6SolutionsCPU > 0) {
                    if (nSK6SolutionsCPU > MAX_SK_PHASE_SIX) {
                        fprintf(stderr, "Warning: Number of phase 6 solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                        nSK6SolutionsCPU = MAX_SK_PHASE_SIX;
                    }

                    //printf("        # Slide Kick Routes: %d\n", nSK6SolutionsCPU);

                    nBlocks = (nUpwarpSolutionsCPU + o->nThreads - 1) / o->nThreads;

                    cudaMemcpyToSymbol(nSKUWSolutions, &nSKUWSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

                    reset_ranges<<<1, 1>>>();
                    find_sk_upwarp_solutions<<<nBlocks, o->nThreads>>>();

                    cudaMemcpyFromSymbol(&nSKUWSolutionsCPU, nSKUWSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                }

                if (nSKUWSolutionsCPU > 0) {
                    if (nSKUWSolutionsCPU > MAX_SK_UPWARP_SOLUTIONS) {
                        fprintf(stderr, "Warning: Number of slide kick upwarp solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                        nSKUWSolutionsCPU = MAX_SK_UPWARP_SOLUTIONS;
                    }

                    //printf("        # Slide Kick Upwarp Solutions: %d\n", nSKUWSolutionsCPU);

                    generate_strain_setups<<<1, 1>>>();

                    cudaMemcpyFromSymbol(&nStrainSetupsCPU, nStrainSetups, sizeof(int), 0, cudaMemcpyDeviceToHost);

                    if (nStrainSetupsCPU > MAX_STRAIN_SETUPS) {
                        fprintf(stderr, "Warning: Number of strain setups for this normal has been exceeded. No more setups for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                        nStrainSetupsCPU = MAX_STRAIN_SETUPS;
                    }

                    nBlocks = (((long long int)nSKUWSolutionsCPU*(long long int)nStrainSetupsCPU) + o->nThreads - 1) / o->nThreads;

                    cudaMemcpyToSymbol(nSpeedSolutions, &nSpeedSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

                    find_speed_solutions<<<nBlocks, o->nThreads>>>();

                    cudaMemcpyFromSymbol(&nSpeedSolutionsCPU, nSpeedSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                }

                if (nSpeedSolutionsCPU > 0) {
                    if (nSpeedSolutionsCPU > MAX_SPEED_SOLUTIONS) {
                        fprintf(stderr, "Warning: Number of speed solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                        nSpeedSolutionsCPU = MAX_SPEED_SOLUTIONS;
                    }

                    //printf("        # Speed Solutions: %d\n", nSpeedSolutionsCPU);

                    nBlocks = (nSpeedSolutionsCPU + o->nThreads - 1) / o->nThreads;

                    cudaMemcpyToSymbol(n10KSolutions, &n10KSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

                    test_speed_solution<<<nBlocks, o->nThreads>>>(p->devSquishEdges, sameNormal ? 4 : 3, p->host_norms[3 * x + 1], uphillAngle, o->maxSlidingSpeed, o->maxSlidingSpeedToPlatform);

                    cudaMemcpyFromSymbol(&n10KSolutionsCPU, n10KSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                }

                if (n10KSolutionsCPU > 0) {
                    if (n10KSolutionsCPU > MAX_10K_SOLUTIONS) {
                        fprintf(stderr, "Warning: Number of 10K solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                        n10KSolutionsCPU = MAX_10K_SOLUTIONS;
                    }

                    printf("---------------------------------------\nTesting Normal: %g, %g, %g\n", startNormal[0], startNormal[1], startNormal[2]);
                    printf("        # 10K Solutions: %d\n", n10KSolutionsCPU);

                    int maxAngleRangeCPU = 0;

                    cudaMemcpyToSymbol(maxAngleRange, &maxAngleRangeCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

                    find_slide_solutions<<<nBlocks, o->nThreads>>>();

                    cudaMemcpyFromSymbol(&maxAngleRangeCPU, maxAngleRange, sizeof(int), 0, cudaMemcpyDeviceToHost);

                    nBlocks = ((maxAngleRangeCPU * n10KSolutionsCPU) + o->nThreads - 1) / o->nThreads;

                    cudaMemcpyToSymbol(nSlideSolutions, &nSlideSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
                    test_slide_angle<<<nBlocks, o->nThreads>>>();
                    cudaMemcpyFromSymbol(&nSlideSolutionsCPU, nSlideSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                }

                if (nSlideSolutionsCPU > 0) {
                    if (nSlideSolutionsCPU > MAX_SLIDE_SOLUTIONS) {
                        fprintf(stderr, "Warning: Number of slide solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                        nSlideSolutionsCPU = MAX_SLIDE_SOLUTIONS;
                    }

                    printf("        # Slide Solutions: %d\n", nSlideSolutionsCPU);

                    nBlocks = (242L * 242L * (long long int)nSlideSolutionsCPU + o->nThreads - 1) / o->nThreads;

                    cudaMemcpyToSymbol(nBDSolutions, &nBDSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

                    find_breakdance_solutions<<<nBlocks, o->nThreads>>>();

                    cudaMemcpyFromSymbol(&nBDSolutionsCPU, nBDSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                }

                if (nBDSolutionsCPU > 0) {
                    if (nBDSolutionsCPU > MAX_BD_SOLUTIONS) {
                        fprintf(stderr, "Warning: Number of breakdance solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                        nBDSolutionsCPU = MAX_BD_SOLUTIONS;
                    }

                    printf("        # Breakdance Solutions: %d\n", nBDSolutionsCPU);

                    nBlocks = (n10KSolutionsCPU + o->nThreads - 1) / o->nThreads;

                    cudaMemcpyToSymbol(nDouble10KSolutions, &nDouble10KSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

                    find_double_10k_solutions<<<nBlocks, o->nThreads>>>();

                    cudaMemcpyFromSymbol(&nDouble10KSolutionsCPU, nDouble10KSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                }

                if (nDouble10KSolutionsCPU > 0) {
                    if (nDouble10KSolutionsCPU > MAX_DOUBLE_10K_SOLUTIONS) {
                        fprintf(stderr, "Warning: Number of double 10K solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                        nDouble10KSolutionsCPU = MAX_DOUBLE_10K_SOLUTIONS;
                    }

                    printf("        # Double 10K Solutions: %d\n", nDouble10KSolutionsCPU);

                    nBlocks = (3 * nDouble10KSolutionsCPU + o->nThreads - 1) / o->nThreads;

                    cudaMemcpyToSymbol(nBullyPushSolutions, &nBullyPushSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

                    find_bully_positions<<<nBlocks, o->nThreads>>>(uphillAngle, o->maxSlidingSpeed, o->maxSlidingSpeedToPlatform);

                    cudaMemcpyFromSymbol(&nBullyPushSolutionsCPU, nBullyPushSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                }

                if (nBullyPushSolutionsCPU > 0) {
                    if (nBullyPushSolutionsCPU > MAX_BULLY_PUSH_SOLUTIONS) {
                        fprintf(stderr, "Warning: Number of bully push solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                        nBullyPushSolutionsCPU = MAX_BULLY_PUSH_SOLUTIONS;
                    }

                    printf("        # Bully Push Solutions: %d\n", nBullyPushSolutionsCPU);

                    if (wf.is_open()) {
                        write_solutions_to_file(startNormal, o, p, x, wf);
                    } else {
                        fprintf(stderr, "Warning: ofstream is not open. No solutions will be written to the output file.\n");
                    }

                    foundSolution = true;
                }
            }
        }
    }

    return foundSolution;
}

int initialise_fst_vars(struct FSTData* p) {
    int cudaError = 0;

    init_reverse_atanG<<<1, 1>>>();
    init_camera_angles<<<1, 1>>>();
    init_mag_set<<<1, 1>>>();
    initialise_floors<<<1, 1>>>();

    cudaError |= init_solution_structs(&(p->s));

    cudaError |= cudaMalloc((void**)&(p->devSquishSpots), 8 * MAX_SQUISH_SPOTS * sizeof(float));
    cudaError |= cudaMalloc((void**)&(p->devNSquishSpots), 4 * sizeof(int));

    init_squish_spots<<<1, 1>>>(p->devSquishSpots, p->devNSquishSpots);

    cudaError |= cudaMalloc((void**)&(p->devStrainSetups), MAX_STRAIN_SETUPS * sizeof(StrainSetup));
    init_strain_setups<<<1, 1>>>(p->devStrainSetups);

    p->host_tris = (short*)std::malloc(36 * sizeof(short));
    p->host_norms = (float*)std::malloc(12 * sizeof(float));

    cudaError |= cudaMalloc((void**)&(p->dev_tris), 36 * sizeof(short));
    cudaError |= cudaMalloc((void**)&(p->dev_norms), 12 * sizeof(float));

    p->host_ceiling_tris = (short*)std::malloc(108 * sizeof(short));
    p->host_ceiling_norms = (float*)std::malloc(36 * sizeof(float));

    cudaError |= cudaMalloc((void**)&(p->dev_ceiling_tris), 108 * sizeof(short));
    cudaError |= cudaMalloc((void**)&(p->dev_ceiling_norms), 36 * sizeof(float));

    p->floorPoints = (short*)std::malloc(4 * 3 * sizeof(short));
    cudaError |= cudaMalloc((void**)&(p->devFloorPoints), 4 * 3 * sizeof(short));

    p->squishEdges = (int*)std::malloc(4 * sizeof(int));
    cudaError |= cudaMalloc((void**)&(p->devSquishEdges), 4 * sizeof(int));

    if (cudaError != 0) {
        if (cudaError | 0x2) {
            fprintf(stderr, "Error: GPU memory allocation failed due to insufficient memory.\n");
            fprintf(stderr, "       It is recommended that you decrease the size of the\n");
            fprintf(stderr, "       reserved memory used for storing sub-solutions.\n");
            //fprintf(stderr, "       Run this program with --help for details.\n");
        }
        else {
            fprintf(stderr, "Error: GPU memory allocation failed with error code: %d.\n", cudaError);
        }
    }

    return cudaError;
}

void free_fst_vars(struct FSTData* p) {
    std::free(p->host_tris);
    std::free(p->host_norms);
    std::free(p->host_ceiling_tris);
    std::free(p->host_ceiling_norms);
    std::free(p->floorPoints);
    std::free(p->squishEdges);
    cudaFree(p->dev_tris);
    cudaFree(p->dev_norms);
    cudaFree(p->dev_ceiling_tris);
    cudaFree(p->dev_ceiling_norms);
    cudaFree(p->devFloorPoints);
    cudaFree(p->devSquishEdges);
    cudaFree(p->devSquishSpots);
    cudaFree(p->devNSquishSpots);
    free_solution_pointers(&(p->s));
}

__global__ void cuda_print_success() {
    printf("CUDA code completed successfully.\n");
}

void print_success() {
    cuda_print_success<<<1, 1>>>();
}
