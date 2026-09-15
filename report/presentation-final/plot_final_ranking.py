#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


# Objective values for all 39 teams in the source report.
# Other team names are intentionally omitted from the presentation artifact.
OBJECTIVES = np.array(
    [
        [1_569_964, 15_112_677, 3_950_969, 3_238_153, 4_354_002, 737_120, 12_950_147, 12_072_285],
        [1_941_056, 15_275_645, 4_162_717, 3_536_654, 4_812_910, 792_055, 12_244_250, 14_449_265],
        [1_794_442, 16_415_669, 4_265_813, 3_288_730, 5_171_244, 794_938, 14_365_981, 14_916_568],
        [2_278_187, 15_769_733, 4_757_773, 3_529_929, 5_056_276, 882_278, 13_910_996, 13_670_227],
        [1_897_010, 16_143_560, 4_202_748, 3_650_067, 5_673_249, 888_748, 14_151_158, 13_877_997],
        [1_918_636, 16_006_971, 4_528_732, 3_870_315, 5_676_615, 841_883, 14_928_193, 14_542_226],
        [2_091_873, 18_128_838, 4_198_662, 4_295_253, 5_574_193, 880_139, 13_923_793, 16_603_238],
        [2_588_411, 17_008_350, 5_324_954, 3_789_908, 5_257_000, 925_741, 14_886_534, 15_627_434],
        [1_969_355, 19_149_216, 5_071_939, 4_014_044, 5_580_523, 887_042, 15_651_861, 14_511_225],
        [1_983_143, 16_647_535, 4_652_063, 3_920_854, 5_053_471, 984_912, 15_782_901, 17_482_731],
        [2_146_041, 17_934_388, 4_845_563, 4_393_961, 5_794_081, 941_882, 16_685_005, 16_307_747],
        [2_279_940, 16_470_531, 4_830_980, 4_674_933, 6_032_678, 973_055, 16_582_905, 15_868_600],
        [2_387_057, 18_663_808, 5_047_507, 4_492_943, 6_039_400, 985_836, 15_154_505, 16_564_175],
        [2_765_810, 17_956_691, 5_598_094, 4_200_264, 5_815_212, 903_197, 16_056_537, 15_660_653],
        [1_876_650, 19_342_382, 4_742_482, 4_803_071, 6_178_446, 947_279, 16_576_707, 16_863_126],
        [2_881_884, 18_568_523, 5_109_301, 4_817_053, 5_736_120, 928_388, 15_988_649, 15_650_895],
        [2_869_491, 17_835_367, 4_910_688, 4_586_498, 6_100_015, 950_368, 15_660_559, 17_431_161],
        [2_648_470, 18_554_176, 5_315_725, 5_006_484, 5_983_703, 953_250, 16_258_691, 16_486_552],
        [2_602_960, 17_907_651, 5_738_175, 4_490_539, 6_077_299, 975_680, 16_507_864, 17_210_469],
        [2_346_760, 20_155_214, 5_049_767, 4_080_351, 6_630_829, 1_032_987, 15_752_477, 18_616_529],
        [2_525_658, 19_529_400, 5_299_968, 4_700_498, 6_211_602, 1_008_392, 16_714_985, 17_940_685],
        [2_842_533, 18_563_430, 6_220_756, 4_904_264, 5_761_183, 986_707, 18_125_705, 17_095_555],
        [3_015_562, 18_955_135, 6_052_948, 4_597_148, 6_316_669, 967_048, 16_343_493, 17_381_674],
        [2_589_496, 19_132_818, 5_781_825, 5_015_103, 5_853_800, 997_443, 18_001_075, 17_543_999],
        [2_249_420, 19_010_094, 5_341_061, 5_197_685, 6_541_486, 1_037_028, 18_026_077, 17_494_310],
        [2_657_157, 21_440_281, 5_313_627, 4_393_111, 7_033_722, 1_011_201, 17_758_664, 17_512_028],
        [2_759_158, 18_746_634, 5_642_876, 5_249_212, 6_586_961, 1_048_580, 17_685_258, 18_795_493],
        [3_274_727, 19_805_358, 6_076_301, 5_088_441, 5_931_876, 1_023_471, 18_191_099, 17_217_985],
        [2_620_018, 19_586_654, 5_432_519, 5_329_567, 6_251_166, 1_076_509, 19_712_505, 18_697_999],
        [3_216_070, 19_078_936, 6_027_672, 5_785_934, 6_768_485, 1_042_937, 17_659_053, 18_137_565],
        [2_501_622, 20_675_130, 5_487_133, 5_934_297, 7_829_240, 1_037_337, 20_466_033, 18_421_979],
        [3_429_398, 18_914_029, 6_020_899, 5_275_002, 6_027_519, 1_049_009, 158_849_689, 19_970_627],
        [2_734_010, 22_695_200, 6_017_629, 4_897_175, 7_397_585, 1_072_079, 19_698_848, 20_046_368],
        [2_748_115, 20_837_319, 5_923_261, 5_487_620, 7_138_225, 1_061_427, 20_858_394, 19_267_710],
        [3_226_583, 19_524_650, 5_300_929, 6_097_478, 7_200_416, 1_103_842, 17_969_344, 21_558_815],
        [2_553_169, 23_183_001, 5_807_308, 5_948_375, 7_451_051, 1_118_101, 21_089_261, 19_813_160],
        [2_969_511, 21_088_773, 6_252_819, 5_529_245, 6_908_242, 1_142_630, 20_504_159, 18_902_927],
        [2_928_920, 22_995_536, 6_361_946, 5_801_478, 6_617_715, 1_062_583, 26_993_435, 19_830_042],
        [3_321_781, 25_424_823, 6_546_873, 5_673_860, 8_639_086, 1_101_634, 20_331_857, 21_896_864],
    ],
    dtype=float,
)

ACCENT = "#4389DC"
SECONDARY = "#81878D"

TEXT = "#3F3F3F"



def main() -> None:
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    other_best = OBJECTIVES[1:].min(axis=0)
    tishii24_indices = OBJECTIVES[0] / other_best * 100
    differences = 100 - tishii24_indices

    font = FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
    figure, axis = plt.subplots(figsize=(13.333, 7.5), dpi=240)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("none")

    x = np.arange(OBJECTIVES.shape[1])
    tishii24_bars = axis.bar(
        x,
        tishii24_indices,
        width=0.52,
        label="tishii24",
        color=ACCENT,
        edgecolor="none",
    )
    axis.axhline(
        100,
        color=SECONDARY,
        linewidth=1.5,
        linestyle=(0, (5, 4)),
    )
    axis.text(
        0.05,
        100.6,
        "Best other finalist = 100",
        ha="left",
        va="bottom",
        fontproperties=font,
        fontsize=12,
        color=SECONDARY,
    )

    axis.set_ylim(80, 110)
    axis.set_yticks(np.arange(80, 111, 5))
    axis.set_ylabel(
        "Objective  (lower is better)",
        fontproperties=font,
        fontsize=17,
        color=TEXT,
        labelpad=14,
    )
    axis.set_xticks(x)
    axis.set_xticklabels(
        [f"P{index}" for index in range(1, OBJECTIVES.shape[1] + 1)],
        fontproperties=font,
        fontsize=15,
        color=TEXT,
    )

    for bar, difference in zip(tishii24_bars, differences):
        center = bar.get_x() + bar.get_width() / 2
        value = bar.get_height()
        axis.annotate(
            "",
            xy=(center, value),
            xytext=(center, 100),
            arrowprops={"arrowstyle": "->", "color": TEXT, "linewidth": 1.5},
        )
        axis.text(
            center,
            (value + 100) / 2,
            f"{abs(difference):.1f}%",
            va="center",
            ha="center",
            fontproperties=font,
            fontsize=11,
            fontweight="bold",
            color=TEXT,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )


    axis.tick_params(axis="x", length=0, pad=10)
    axis.tick_params(axis="y", colors=TEXT, labelsize=12, length=0)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=1,
        frameon=False,
        prop=font,
        fontsize=14,
    )
    for spine in axis.spines.values():
        spine.set_visible(False)

    figure.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.1)

    figure.savefig(output_dir / "final-objective-by-problem.png", dpi=240, transparent=True)
    figure.savefig(output_dir / "final-objective-by-problem.svg", transparent=False)
    plt.close(figure)

    print(f"problems_won={(differences > 0).sum()}/{len(differences)}")
    print(f"mean_reduction_percent={differences.mean():.3f}")


if __name__ == "__main__":
    main()
