from .core import StringIntEnum

# from a standard atlas https://github.com/neurodata/neuroparc/blob/master/atlases/label/Human/Anatomical-labels-csv/Glasser.csv
# with a few custmized region
class HomoSapiens(StringIntEnum):
    primary_visual_cortex = 0
    medial_superior_temporal_area = 1
    sixth_visual_area = 2
    second_visual_area = 3
    third_visual_area = 4
    forth_visual_area = 5
    eighth_visual_area = 6
    primary_motor_cortex = 7
    primary_sensory_cortex = 8
    frontal_eye_field = 9
    premotor_eye_field = 10
    area_55b = 11
    area_v3a = 12
    retrosplenial_complex = 13
    parietooccipital_sulcus_area_2 = 14
    seventh_visual_area = 15
    intraparietal_sulcus_area_1 = 16
    fusiform_face_complex = 17
    area_v3b = 18
    lateral_occipital_area_1 = 19
    lateral_occipital_area_2 = 20
    posterior_inferotemporal_complex = 21
    middle_temporal_area = 22
    primary_auditory_cortex = 23
    perisylvian_language_area = 24
    superior_frontal_language_area = 25
    precuneus_visual_area = 26
    superior_temporal_visual_area = 27
    medial_area_7p = 28
    area_7m = 29
    parietooccipital_sulcus_area_1 = 30
    area_23d = 31
    area_ventral_23_a_b = 32
    area_dorsal_23_a_b = 33
    area_31p_ventral = 34
    area_5m = 35
    area_5m_ventral = 36
    area_23c = 37
    area_5l = 38
    dorsal_area_24d = 39
    lateral_area_7a = 40
    supplementary_and_cingulate_eye_field = 41
    area_6m_anterior = 42
    medial_6m_anterior = 43
    medial_area_7a = 44
    lateral_area_7p = 45
    area_7pc = 46
    area_lateral_intraparietal_ventral = 47
    ventral_intraparietal_complex = 48
    medial_intraparietal_area = 49
    area_1 = 50
    area_2 = 51
    area_3a = 52
    dorsal_area_6 = 53
    area_6mp = 54
    ventral_area_6 = 55
    area_posterior_24 = 56
    area_33_prime = 57
    anterior_24_prime = 58
    area_p32_prime = 59
    area_a24 = 60
    area_dorsal_32 = 61
    area_8bm = 62
    area_p32 = 63
    area_10r = 64
    area_47m = 65
    area_8av = 66
    area_8ad = 67
    area_9_middle = 68
    area_8b_lateral = 69
    area_9_posterior = 70
    area_10d = 71
    area_8c = 72
    area_44 = 73
    area_45 = 74
    area_47l = 75
    area_anterior_47r = 76
    rostral_area_6 = 77
    area_ifja = 78
    area_ifjp = 79
    area_ifsp = 80
    area_ifsa = 81
    area_posterior_9_46v = 82
    area_46 = 83
    area_anterior_9_46v = 84
    area_9_46d = 85
    area_9_anterior = 86
    area_10v = 87
    area_anterior_10p = 88
    polar_10p = 89
    area_11 = 90
    area_13 = 91
    orbital_frontal_complex = 92
    area_47s = 93
    area_lateral_intraparietal_dorsal = 94
    area_6_anterior = 95
    inferior_6_8_transitional_area = 96
    superior_6_8_transitional_area = 97
    area_43 = 98
    area_op4_pv = 99
    area_op1_sii = 100
    area_op2_3_vs = 101
    area_52 = 102
    retroinsular_cortex = 103
    area_pfcm = 104
    posterior_insular_area_2 = 105
    area_ta2 = 106
    frontal_opercular_area_4 = 107
    middle_insular_area = 108
    pirform_cortex = 109
    anterior_ventral_insular_area = 110
    anterior_agranular_insula_complex = 111
    frontal_opercular_area_1 = 112
    frontal_opercular_area_3 = 113
    frontal_opercular_area_2 = 114
    area_pft = 115
    anterior_intraparietal_area = 116
    entorhinal_cortex = 117
    presubiculum = 118
    hippocampus = 119
    prostraite_area = 120
    perirhinal_ectorhinal_cortex = 121
    area_stga = 122
    parabelt_complex = 123
    auditory_5_complex = 124
    parahippocampal_area_1 = 125
    parahippocampal_area_3 = 126
    area_stsd_anterior = 127
    area_stsd_posterior = 128
    area_stsv_posterior = 129
    area_tg_dorsal = 130
    area_te1_anterior = 131
    area_te1_posterior = 132
    area_te2_anterior = 133
    area_tf = 134
    area_te2_posterior = 135
    area_pht = 136
    area_ph = 137
    area_temporoparietooccipital_junction_1 = 138
    area_temporoparietooccipital_junction_2 = 139
    area_temporoparietooccipital_junction_3 = 140
    dorsal_transitional_visual_area = 141
    area_pgp = 142
    area_intraparietal_2 = 143
    area_intraparietal_1 = 144
    area_intraparietal_0 = 145
    area_pf_opercular = 146
    area_pf_complex = 147
    area_pfm_complex = 148
    area_pgi = 149
    area_pgs = 150
    area_v6a = 151
    ventromedial_visual_area_1 = 152
    ventromedial_visual_area_3 = 153
    parahippocampal_area_2 = 154
    area_v4t = 155
    area_fst = 156
    area_v3cd = 157
    area_lateral_occipital_3 = 158
    ventromedial_visual_area_2 = 159
    area_31pd = 160
    area_31a = 161
    ventral_visual_complex = 162
    area_25 = 163
    area_s32 = 164
    posterior_ofc_complex = 165
    area_posterior_insular_1 = 166
    insular_granular_complex = 167
    area_frontal_opercular_5 = 168
    area_posterior_10p = 169
    area_posterior_47r = 170
    area_tg_ventral = 171
    medial_belt_complex = 172
    lateral_belt_complex = 173
    auditory_4_complex = 174
    area_stsv_anterior = 175
    area_te1_middle = 176
    parainsular_area = 177
    area_anterior_32_prime = 178
    
    # customized 
    ventral_sensorimotor_cortex = 179