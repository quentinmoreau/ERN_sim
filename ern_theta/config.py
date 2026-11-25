"""Paths, event codes and subject lists for the #EEGManyAnalysts Flanker data.

    $ERN_THETA_DATA/
        ernpipeline/<subject>/<subject>_desc-epochs_eeg.fif       # response-locked
        cong_incong_epochs/<subject>_target_cong_incong-epo.fif   # stimulus-locked
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get('ERN_THETA_DATA', REPO_ROOT / 'data'))
RESULTS_DIR = Path(os.environ.get('ERN_THETA_RESULTS', REPO_ROOT / 'results'))

ERN_EPOCHS_DIR = DATA_DIR / 'ernpipeline'
N2_EPOCHS_DIR = DATA_DIR / 'cong_incong_epochs'

# Optional: lagged Hilbert autocoherence code, used only by the supplementary notebooks
LHAC_DIR = Path(os.environ.get('LHAC_DIR', REPO_ROOT / 'lagged_hilbert_autocoherence' / 'python'))


def ern_epochs_file(subject):
    return ERN_EPOCHS_DIR / subject / f'{subject}_desc-epochs_eeg.fif'


def n2_epochs_file(subject):
    return N2_EPOCHS_DIR / f'{subject}_target_cong_incong-epo.fif'


CHANNEL = 'FZ'
MIN_TRIALS = 5  # subjects/conditions with fewer trials are skipped

# Event codes in the epochs files
RESPONSE_CODES = {
    'incorrect': ['108', '109', '118', '119', '128', '129', '138', '139'],
    'correct':   ['106', '107', '116', '117', '126', '127', '136', '137'],
}
STIMULUS_CODES = {
    'incongruent': ['104'],
    'congruent':   ['134'],
}

# Subjects included in the response-locked (ERN) analysis
ERN_SUBJECTS = [
    'sub-AG05AS29', 'sub-AH17AR05', 'sub-AL07NK06', 'sub-AL08AS02', 'sub-AR06AS06',
    'sub-AR10IO04', 'sub-AT06ED25', 'sub-AU09GE22', 'sub-CH04ER19', 'sub-CH05NK28',
    'sub-CH06UT07', 'sub-CH07AD15', 'sub-CH09AS13', 'sub-CK04NS18', 'sub-CK06ED08',
    'sub-CK06ST11', 'sub-CK06US15', 'sub-DADARE11', 'sub-DT05EV26', 'sub-DT07ES22',
    'sub-DTINTZ10', 'sub-EK05AN14', 'sub-EK06ER26', 'sub-EK07CH24', 'sub-EL05JO26',
    'sub-EL06LD17', 'sub-EL07EL14', 'sub-EN04ER01', 'sub-EN05AN02', 'sub-EN05KT02',
    'sub-EN05US05', 'sub-EN06AN21', 'sub-EN08CH27', 'sub-EP05EN09', 'sub-ER03ER09',
    'sub-ER04ER12', 'sub-ER04RT24', 'sub-ER05AN13', 'sub-ER05AV01', 'sub-ER05EL09',
    'sub-ER05EL26', 'sub-ER05ER07', 'sub-ER05IM29', 'sub-ER05MA17', 'sub-ER05NG11',
    'sub-ER05RK06', 'sub-ER05RT24', 'sub-ER06EL06', 'sub-ER06EN27', 'sub-ER06ET15',
    'sub-ER06IS22', 'sub-ER06US24', 'sub-ER07AN28', 'sub-ER07AS24', 'sub-ER07EF15',
    'sub-ER07EL10', 'sub-ER07ER23', 'sub-ER07ER30', 'sub-ER07RL02', 'sub-ER08AS28',
    'sub-ER08EN05', 'sub-ER09NK30', 'sub-ES06ER27', 'sub-ES06NK11', 'sub-EZ04SE15',
    'sub-GA04LI18', 'sub-HEERLD19', 'sub-IA05NS20', 'sub-ID07EN28', 'sub-IG06AI28',
    'sub-IG06LF01', 'sub-IG06NS07', 'sub-IG06WE03', 'sub-IG07RT16', 'sub-IK06ET06',
    'sub-IK09LD05', 'sub-IN05ED23', 'sub-IN06AS14', 'sub-IS06RG09', 'sub-IT08ND26',
    'sub-JU05IN24', 'sub-KA04LF11', 'sub-KE05AN23', 'sub-KE07AN05', 'sub-KE07ER04',
    'sub-KE08AS23', 'sub-KN05EN09', 'sub-KO07EL16', 'sub-LA08RE26', 'sub-LO06RE10',
    'sub-LY07EL23', 'sub-MA06LO21', 'sub-ME05NS04', 'sub-NA04AV25', 'sub-ND05NG06',
    'sub-ND06EF01', 'sub-NE06AS01', 'sub-NE06EL01', 'sub-NG03YI17', 'sub-NG06EA08',
    'sub-NK06AS24', 'sub-NK06RG30', 'sub-NN04AN24', 'sub-NN05AN05', 'sub-NN05IR17',
    'sub-NN06TH16', 'sub-NN07EN14', 'sub-NN07ID18', 'sub-NN08AS03', 'sub-NN08EL29',
    'sub-NN08ER22', 'sub-NN08LM06', 'sub-NN09LM18', 'sub-NN09NS31', 'sub-NO05RT22',
    'sub-NS09EL05', 'sub-NY04DT26', 'sub-NZ04EN16', 'sub-NZ09ER04', 'sub-OH05NK29',
    'sub-ON05EL22', 'sub-OW07RK26', 'sub-PP07AS29', 'sub-RASUAN07', 'sub-RF09EL18',
    'sub-RG05RS16', 'sub-RG06AS09', 'sub-RO06AN18', 'sub-RT07LD06', 'sub-RYRAAN26',
    'sub-RZ05ND19', 'sub-SA05AH22', 'sub-SE05AS26', 'sub-SE05ZI24', 'sub-SE06AM29',
    'sub-SE06NS17', 'sub-ST03ED03', 'sub-ST07NK15', 'sub-TA03RL11', 'sub-TE05AS09',
    'sub-TE05NS01', 'sub-TE10AN12', 'sub-TH07AR12', 'sub-TH08ER21', 'sub-TT06LF28',
    'sub-TT08NS17', 'sub-TZ08US31', 'sub-US07US10', 'sub-US08KE27', 'sub-VA05AS27',
    'sub-VA05OR26', 'sub-VA06NZ29', 'sub-VA06OV12', 'sub-WA04EJ22', 'sub-WA05EJ08',
    'sub-XX04AS28',
]

# Subjects included in the stimulus-locked (N2, congruent vs incongruent) analysis
N2_SUBJECTS = [
    'sub-AH17AR05', 'sub-AI09ID16', 'sub-AL07NK06', 'sub-AL08AS02', 'sub-AR06AS06',
    'sub-AR10IO04', 'sub-AT06ED25', 'sub-AU09GE22', 'sub-CH04ER19', 'sub-CH05NK28',
    'sub-CH06UT07', 'sub-CH07AD15', 'sub-CH07NK17', 'sub-CH07RK18', 'sub-CH10NG24',
    'sub-CK04NS18', 'sub-CK05EN08', 'sub-CK06ED08', 'sub-CK06US15', 'sub-CK07NK21',
    'sub-DADARE11', 'sub-DS08NK08', 'sub-DT05AH09', 'sub-DT05EV26', 'sub-DT07ES22',
    'sub-DTINTZ10', 'sub-EA06ER02', 'sub-EK05AN14', 'sub-EK06ER26', 'sub-EK07CH24',
    'sub-EL04AM03', 'sub-EL05JO26', 'sub-EL06EL11', 'sub-EL06LD17', 'sub-EL06ST21',
    'sub-EL07EL14', 'sub-EN04ER01', 'sub-EN04ER14', 'sub-EN05AN02', 'sub-EN05ED14',
    'sub-EN05KT02', 'sub-EN05US05', 'sub-EN06AN21', 'sub-EN08CH27', 'sub-EP05EN09',
    'sub-ER03ER09', 'sub-ER03LF17', 'sub-ER04CH03', 'sub-ER04ER12', 'sub-ER04RT24',
    'sub-ER04ST03', 'sub-ER05AN13', 'sub-ER05AR27', 'sub-ER05AS20', 'sub-ER05AV01',
    'sub-ER05EL09', 'sub-ER05EL26', 'sub-ER05ER07', 'sub-ER05IM29', 'sub-ER05MA17',
    'sub-ER05NG11', 'sub-ER05RI09', 'sub-ER05RK06', 'sub-ER05RT24', 'sub-ER06AN07',
    'sub-ER06AN28', 'sub-ER06EF14', 'sub-ER06EL06', 'sub-ER06EN03', 'sub-ER06EN27',
    'sub-ER06ET15', 'sub-ER06IM15', 'sub-ER06IS22', 'sub-ER06KR20', 'sub-ER06NK09',
    'sub-ER06US24', 'sub-ER07AS24', 'sub-ER07EF15', 'sub-ER07EN04', 'sub-ER07ER23',
    'sub-ER07ER30', 'sub-ER07ND12', 'sub-ER07RL02', 'sub-ER08AH23', 'sub-ER08AS28',
    'sub-ER08EN05', 'sub-ER08ER03', 'sub-ER08RT29', 'sub-ER09DO22', 'sub-ER09NK30',
    'sub-ER09NZ11', 'sub-ER10IG12', 'sub-ES06ER27', 'sub-ES06NK11', 'sub-EV06AN16',
    'sub-EZ04SE15', 'sub-FA04HU23', 'sub-FE06AN24', 'sub-GA04LI18', 'sub-GA06OF10',
    'sub-GY06LO26', 'sub-HEERLD19', 'sub-HL09AN11', 'sub-HR07ED31', 'sub-IA05NS20',
    'sub-ID07EN28', 'sub-IG06LF01', 'sub-IG06NS07', 'sub-IG06WE03', 'sub-IG07RT16',
    'sub-IK06ET06', 'sub-IK09LD05', 'sub-IN05ED23', 'sub-IN06AS14', 'sub-IN07ET30',
    'sub-IR04ET20', 'sub-IS06RG09', 'sub-IT08ND26', 'sub-JA06AS11', 'sub-JU05IN24',
    'sub-KA04LF11', 'sub-KE04NI10', 'sub-KE05AN23', 'sub-KE06EL16', 'sub-KE07AN05',
    'sub-KE07ER04', 'sub-KE08AS23', 'sub-KI05US17', 'sub-KN05EN09', 'sub-KO07EL16',
    'sub-LA08RE26', 'sub-LE04ON09', 'sub-LF04AS01', 'sub-LF07RT29', 'sub-LO06RE10',
    'sub-LY07EL23', 'sub-MA06LO21', 'sub-ME05NS04', 'sub-MI07AH25', 'sub-NA04AV25',
    'sub-NA04OV25', 'sub-ND05NG06', 'sub-ND06EF01', 'sub-NE06AS01', 'sub-NE06AS31',
    'sub-NE06EL01', 'sub-NG03YI17', 'sub-NG05AS30', 'sub-NG06NI21', 'sub-NG06UL04',
    'sub-NG07AN22', 'sub-NK06AS24', 'sub-NK06RG30', 'sub-NN04AN24', 'sub-NN05AN05',
    'sub-NN05IR17', 'sub-NN06TH16', 'sub-NN06US23', 'sub-NN07EN14', 'sub-NN07ID18',
    'sub-NN08AS03', 'sub-NN08EL29', 'sub-NN08ER22', 'sub-NN08LM06', 'sub-NN09LM18',
    'sub-NN09NS31', 'sub-NO05RT22', 'sub-NS05RO15', 'sub-NS09EL05', 'sub-NY04DT26',
    'sub-NZ04EN16', 'sub-NZ09ER04', 'sub-OH05NK29', 'sub-ON05EL22', 'sub-ON06US07',
    'sub-OW07RK26', 'sub-PP07AS29', 'sub-RASUAN07', 'sub-RF09EL18', 'sub-RG05AS08',
    'sub-RG06AS09', 'sub-RG07US04', 'sub-RN10AN23', 'sub-RS06RS22', 'sub-RS07LF20',
    'sub-RT04KE01', 'sub-RT05EL10', 'sub-RT06AN02', 'sub-RT07LD06', 'sub-RZ05ND19',
    'sub-SE05EN28', 'sub-SE05ZI24', 'sub-SE06AM29', 'sub-SE06NS17', 'sub-SS04PH05',
    'sub-ST03ED03', 'sub-ST07NK15', 'sub-TA03RL11', 'sub-TE03AS11', 'sub-TE05AS09',
    'sub-TE05NS01', 'sub-TE08IO29', 'sub-TE10AN12', 'sub-TH07AR12', 'sub-TL08AN16',
    'sub-TT08NS17', 'sub-TZ08US31', 'sub-UN07AS25', 'sub-US05RK06', 'sub-US07US10',
    'sub-US08KE27', 'sub-VA03ER17', 'sub-VA05AS27', 'sub-VA05ER28', 'sub-VA05OR26',
    'sub-VA06EY24', 'sub-VA06NZ29', 'sub-VA06OV12', 'sub-WA04EJ22', 'sub-WA05EJ08',
    'sub-XX04AS28',
]
