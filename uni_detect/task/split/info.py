IDMT_AIR_MAP = {
    'tubeleak_noleak': 0,
    'tubeleak_leak': 1,
    'ventleak_noleak': 2,
    'ventleak_leak': 3,
    'ventlow_noleak': 4,
    'ventlow_leak': 5
}

WT_PG_MAP = {
    'broken': 0,
    'healthy': 1,
    'missing_tooth': 2,
    'root_crack': 3,
    'wear': 4
}

MaFaulDa_DS_MAP = {
    'horizontal-misalignment': 0,
    'imbalance': 1,
    'normal': 2,
    'overhang_ball_fault': 3,
    'overhang_cage_fault': 4,
    'overhang_outer_race': 5,
    'underhang_ball_fault': 6,
    'underhang_cage_fault': 7,
    'underhang_outer_race': 8,
    'vertical-misalignment': 9,
}

SD_B_MAP = {
    'IF0.2': 0,
    'IF0.4': 1,
    'IF0.6': 2,
    'NC': 3,
    'OF0.2': 4,
    'OF0.4': 5,
    'OF0.6': 6,
    'RF0.2': 7,
    'RF0.4': 8,
    'RF0.6': 9
}

SD_G_MAP = {
    'normal': 0,
    'planetrayfracture': 1,
    'planetraypitting': 2,
    'planetraywear': 3,
    'sunfracture': 4,
    'sunpitting': 5,
    'sunwear': 6
}

UMGED_G_MAP = {
    'G1': 0,
    'G2': 1
}

UMGED_E_MAP = {
    'E00': 0,
    'E02': 1,
    'E04': 2,
    'E06': 3,
    'E08': 4,
    'E10': 5,
    'E12': 6,
    'E14': 7,
    'E16': 8,
    'E18': 9,
    'E20': 10,
}

PU_MAP = {
    'healthy': 0,
    'IR': 1,
    'OR': 2
}


SPLIT_DS_MAP = {
    'idmt_air': IDMT_AIR_MAP,
    'wt_plane_gearbox': WT_PG_MAP,
    'mafaulda_vib': MaFaulDa_DS_MAP,
    'mafaulda_sound': MaFaulDa_DS_MAP,
    'mafaulda_vib_csv': MaFaulDa_DS_MAP,
    'mafaulda_sound_csv': MaFaulDa_DS_MAP,
    'sdust_bearing': SD_B_MAP,
    'sdust_gear': SD_G_MAP,
    'umged_G': UMGED_G_MAP,
    'umged_E': UMGED_E_MAP,
    'umged_sound': UMGED_E_MAP,
    'umged_cur': UMGED_E_MAP,
    'umged_vib': UMGED_E_MAP,
    'umged_vol': UMGED_E_MAP,
    'pu_cur': PU_MAP,
    'pu_vib': PU_MAP,
}
